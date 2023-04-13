from transformers import BertPreTrainedModel, BertModel
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch
from transformers.modeling_outputs import TokenClassifierOutput 
from utils import *

class CustomizedTokenClassification(BertPreTrainedModel):


    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.args = getArgs()
        if self.args.lstm > 0:
            self.classifier = nn.Linear(2 * self.args.lstm_dimension, config.num_labels)
            self.birnn = nn.LSTM(config.hidden_size+7, self.args.lstm_dimension, num_layers=1, bidirectional=True, batch_first=True)

        #self.crf = CRF(config.num_labels, batch_first=True)
        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if self.args.lstm > 0:
            input_tokens = get_tokens(input_ids.tolist())
            features = [gen_miller_features("".join(tokens)) for tokens in input_tokens]
            sequence_output = outputs[0]
            sequence_output = torch.cat([sequence_output, torch.Tensor(features).cuda()], 2)
            #print ("concat shape:", sequence_output.shape)
            sequence_output, _ = self.birnn(sequence_output)
            sequence_output = self.dropout(sequence_output)
            logits = self.classifier(sequence_output)
        else:
            sequence_output = outputs[0]
            sequence_output = self.dropout(sequence_output)
            logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                if self.args.restrictedLoss > 0:
                    cd3_mask = active_labels != -100
                    new_active_labels = torch.masked_select(active_labels,cd3_mask)
                    cd3_mask = cd3_mask.unsqueeze(0)
                    #print (cd3_mask)
                    cd3_mask = torch.transpose(cd3_mask, 0, 1)
                    #print (cd3_mask)
                    cd3_mask = cd3_mask.repeat(1,2)
                    #print ("cd3_mask:", cd3_mask)
                    new_active_logits = torch.masked_select(active_logits, cd3_mask).reshape(-1,2)
                    loss = loss_fct(new_active_logits, new_active_labels)
                else:
                    loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        #loss = -1*self.crf(logits, labels, mask=attention_mask.byte())
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
