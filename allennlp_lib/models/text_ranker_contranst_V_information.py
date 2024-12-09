from typing import List, Dict, Optional, Any

from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder, TimeDistributed
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask, get_range_vector, get_device_of
from allennlp.training.metrics import CategoricalAccuracy
from allennlp_lib.training.metrics import ListCompareEmAndF1, FullRecallRank
from allennlp_lib.models.utils import replace_masked_values_with_big_negative_number

import torch.nn.functional as F

def calculate_v_information(logits, labels, logits_question_only):
    loss_y_given_nothing = -torch.sum(F.log_softmax(logits_question_only, dim=-1) * labels, dim=-1)
    loss_y_given_x = -torch.sum(F.log_softmax(logits, dim=-1) * labels, dim=-1)
    v_info_loss = loss_y_given_nothing.mean() - loss_y_given_x.mean()
    return v_info_loss


def contrastive_loss_func(embeddings, pos_indices, neg_indices, margin=1.3):
  
    batch_size = embeddings.size(0)
    

    total_pos_loss = 0
    total_neg_loss = 0


    for i in range(batch_size):

        pos_embeds = embeddings[i][pos_indices[i]]
        neg_embeds = embeddings[i][neg_indices[i]]
        if pos_embeds.size(0) > 1:
           
            pos_similarity = F.cosine_similarity(pos_embeds.unsqueeze(1), pos_embeds.unsqueeze(0), dim=-1)
            pos_loss = 1 - pos_similarity.mean()
        else:
            pos_loss = 0

        if neg_embeds.size(0) > 0:
                
            neg_similarity = F.cosine_similarity(pos_embeds.unsqueeze(1), neg_embeds.unsqueeze(0), dim=-1)
            neg_loss = F.relu(margin - neg_similarity).mean()
        else:
            neg_loss = 0

        total_pos_loss += pos_loss
        total_neg_loss += neg_loss

    return (total_pos_loss + total_neg_loss) / batch_size


@Model.register("text_ranker_constranst_v_information")
class TextRanker(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        has_single_positive: bool,
        seq2vec_encoder: Seq2VecEncoder = None,
        seq2seq_encoder: Seq2SeqEncoder = None,
        feedforward: FeedForward = None,
        dropout: float = None,
        margin: float = 1.3,  
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder

        self._seq2seq_encoder = TimeDistributed(seq2seq_encoder) if seq2seq_encoder else None
        self._seq2vec_encoder = TimeDistributed(seq2vec_encoder) if seq2vec_encoder else None
        self._feedforward = TimeDistributed(feedforward) if feedforward else None

        if self._feedforward is not None:
            self._ranker_input_dim = feedforward.get_output_dim()
        elif self._seq2vec_encoder is not None:
            self._ranker_input_dim = self._seq2vec_encoder.get_output_dim()
        else:
            self._ranker_input_dim = self._text_field_embedder.get_output_dim()

        self._dropout = torch.nn.Dropout(dropout) if dropout else None

        self._ranker_layer = TimeDistributed(torch.nn.Linear(self._ranker_input_dim, 1))

        self._select_metrics = ListCompareEmAndF1()
        self._rank_metrics = ListCompareEmAndF1()
        self._full_recall_rank_metrics = FullRecallRank()

        self._has_single_positive = has_single_positive
        if self._has_single_positive:
            self._loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        else:
            self._loss = torch.nn.BCEWithLogitsLoss()
        
        self._contrastive_margin = margin  
        initializer(self)

    def forward(
        self,
        texts: TextFieldTensors,
        questions: TextFieldTensors,
        labels: torch.Tensor = None,
        cls_indices: torch.Tensor = None,
        metadata: List[Dict[str, Any]] = None
    ) -> Dict[str, torch.Tensor]:
        
        embedded_texts = self._text_field_embedder(texts, num_wrapping_dims=1)
        texts_mask = get_text_field_mask(texts, num_wrapping_dims=1)

        embedded_questions = self._text_field_embedder(questions, num_wrapping_dims=1)
        questions_mask = get_text_field_mask(questions, num_wrapping_dims=1)
      
        if self._seq2seq_encoder:
            embedded_texts = self._seq2seq_encoder(embedded_texts, mask=texts_mask)
            embedded_questions = self._seq2seq_encoder(embedded_questions, mask=questions_mask)

        if self._seq2vec_encoder:
            embedded_texts = self._seq2vec_encoder(embedded_texts, mask=texts_mask)
            embedded_questions = self._seq2vec_encoder(embedded_questions, mask=questions_mask)
        else:
            range_vector = get_range_vector(embedded_texts.shape[1], get_device_of(embedded_texts))
            embedded_texts = [embedded_text[range_vector, cls_index, :] for embedded_text, cls_index in
                              zip(torch.unbind(embedded_texts, dim=0), torch.unbind(cls_indices, dim=0))]
            embedded_texts = torch.stack(embedded_texts, dim=0)

            range_vector_1 = get_range_vector(embedded_questions.shape[1], get_device_of(embedded_questions))
            embedded_questions = [embedded_question[range_vector_1, cls_index, :] for embedded_question, cls_index in
                          zip(torch.unbind(embedded_questions, dim=0), torch.unbind(cls_indices, dim=0))]
            embedded_questions = torch.stack(embedded_questions, dim=0)

        if self._dropout:
            embedded_texts = self._dropout(embedded_texts)
            embedded_questions = self._dropout(embedded_questions)

        if self._feedforward is not None:
            embedded_texts = self._feedforward(embedded_texts)
            embedded_questions = self._feedforward(embedded_questions)
        logits = self._ranker_layer(embedded_texts).squeeze(-1)
        logits = replace_masked_values_with_big_negative_number(logits, texts_mask[:, :, 0])
        probs = (logits.softmax(dim=-1) if self._has_single_positive else logits.sigmoid())
        token_ids = util.get_token_ids_from_text_field_tensors(texts)

        output_dict = {"token_ids": token_ids, "probs": probs}

        if metadata:
            output_dict["contexts"] = [metadatum.get("contexts", None) for metadatum in metadata]

        if labels is not None:
            output_dict["labels"] = labels.detach().cpu().numpy()

            # Loss
            labels_mask = labels != -1
            logits = replace_masked_values_with_big_negative_number(logits, labels_mask)
            labels[labels == -1] = 0

            if self._has_single_positive:
                labels = labels.nonzero(as_tuple=True)[1]

            if logits.numel():  # because in mixed-answerable setup, we can have 0 contexts (rarely)
                main_loss = self._loss(logits, labels)
                logits_question_only = self._ranker_layer(embedded_questions).squeeze(-1)
                logits_question_only = replace_masked_values_with_big_negative_number(logits_question_only, questions_mask[:, :, 0])
             
                first_values = logits_question_only[:, 0].unsqueeze(1)

            
                logits_question_only = first_values.expand_as(logits_question_only)
                logits_question_only = replace_masked_values_with_big_negative_number(logits_question_only, labels_mask)

         
                v_information_loss = -calculate_v_information(logits, labels, logits_question_only)
         
      
                batch_size, num_docs = labels.size()
                pos_indices = []
                neg_indices = []
                for i in range(batch_size):
                    pos = (labels[i] == 1).nonzero(as_tuple=True)[0]
                    neg = (labels[i] == 0).nonzero(as_tuple=True)[0]
                    pos_indices.append(pos)
                    neg_indices.append(neg)
                contrastive_loss = contrastive_loss_func(embedded_texts, pos_indices, neg_indices)

            else:
                main_loss = sum([p.sum() for p in self.parameters()]) * 0.0  # otherwise pytorch complains.
                v_information_loss = sum([p.sum() for p in self.parameters()]) * 0.0  # otherwise pytorch complains.
                contrastive_loss = sum([p.sum() for p in self.parameters()]) * 0.0  # otherwise pytorch complains.

            # Total loss
            loss = 0.5*main_loss + 0.2*v_information_loss+0.3*contrastive_loss
            output_dict["loss"] = loss

            # Evaluation
            output_dict["predicted_select_indices"] = []
            output_dict["predicted_rank_indices"] = []
            output_dict["predicted_ordered_indices"] = []
            for index in range(embedded_texts.shape[0]):
                if self._has_single_positive:
                    pos_label_indices = labels[index].unsqueeze(0)
                else:
                    pos_label_indices = labels[index].nonzero(as_tuple=True)[0]

                threshold = 0.5
                predicted_indices = (probs[index] > threshold).nonzero(as_tuple=True)[0]
                self._select_metrics(predicted_indices, pos_label_indices)
                output_dict["predicted_select_indices"].append(predicted_indices)

                predicted_indices = probs[index].topk(k=len(pos_label_indices))[1]
                self._rank_metrics(predicted_indices, pos_label_indices)
                output_dict["predicted_rank_indices"].append(predicted_indices)

                predicted_indices = probs[index].sort(descending=True)[1]
                self._full_recall_rank_metrics(predicted_indices, pos_label_indices)
                output_dict["predicted_ordered_indices"].append(predicted_indices)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        exact_match, f1_score = self._rank_metrics.get_metric(reset)
        metrics["rank_em"] = exact_match
        metrics["rank_f1"] = f1_score

        exact_match, f1_score = self._select_metrics.get_metric(reset)
        metrics["select_em"] = exact_match
        metrics["select_f1"] = f1_score

        full_recall_rank = self._full_recall_rank_metrics.get_metric(reset)
        metrics["full_recall_rank"] = full_recall_rank

        return metrics

    default_predictor = "text_ranker_contranst_V_information"
