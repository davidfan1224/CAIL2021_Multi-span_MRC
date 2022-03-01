# from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.configuration_bert import BertConfig
from pytorch_pretrained_bert.modeling_bert import BertLayer, BertPreTrainedModel, BertModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torchcrf import CRF
VERY_NEGATIVE_NUMBER = -1e29


class CailModel(BertPreTrainedModel):
    def __init__(self, config, answer_verification=True, hidden_dropout_prob=0.3, need_birnn=False, rnn_dim=128):
        super(CailModel, self).__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.qa_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_answers = 4  # args.max_n_answers + 1
        self.qa_outputs = nn.Linear(config.hidden_size*4, 2)
        self.qa_classifier = nn.Linear(config.hidden_size, self.num_answers)
        # self.apply(self.init_bert_weights)

        self.answer_verification = answer_verification
        head_num = config.num_attention_heads // 4

        self.coref_config = BertConfig(num_hidden_layers=1, num_attention_heads=head_num,
                                       hidden_size=config.hidden_size, intermediate_size=256 * head_num)

        self.coref_layer = BertLayer(self.coref_config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        out_dim = config.hidden_size
        self.need_birnn = need_birnn
        # 如果为False，则不要BiLSTM层
        if need_birnn:
            self.birnn = nn.LSTM(config.hidden_size, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            self.gru = nn.GRU(config.hidden_size, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            out_dim = rnn_dim * 2
            

        self.hidden2tag = nn.Linear(out_dim, 2)   # I O 二分类
        # self.crf = CRF(config.num_labels, batch_first=True)
        self.crf = CRF(2, batch_first=True)

        self.init_weights()

        if self.answer_verification:
            self.retionale_outputs = nn.Linear(config.hidden_size*4, 1)
            self.unk_ouputs = nn.Linear(config.hidden_size, 1)
            self.doc_att = nn.Linear(config.hidden_size*4, 1)
            self.yes_no_ouputs = nn.Linear(config.hidden_size*4, 2)
            # self.yes_no_ouputs_noAttention = nn.Linear(config.hidden_size, 2)
            self.ouputs_cls_3 = nn.Linear(config.hidden_size*4, 3)

            self.beta = 100
        else:
            # self.unk_yes_no_outputs_dropout = nn.Dropout(config.hidden_dropout_prob)
            self.unk_yes_no_outputs = nn.Linear(config.hidden_size, 3)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None,
                unk_mask=None, yes_mask=None, no_mask=None, answer_masks=None, answer_nums=None, label_ids=None):
        # sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
        #                                            output_all_encoded_layers=True)
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = sequence_output[1]
        sequence_output_IO = sequence_output[-1]
        sequence_output = torch.cat((sequence_output[-4], sequence_output[-3], sequence_output[-2],
                                     sequence_output[-1]), -1)    # 拼接BERT最后四层


        if self.answer_verification:
            batch_size = sequence_output.size(0)
            seq_length = sequence_output.size(1)
            hidden_size = sequence_output.size(2)
            sequence_output_matrix = sequence_output.view(batch_size*seq_length, hidden_size)
            rationale_logits = self.retionale_outputs(sequence_output_matrix)
            rationale_logits = F.softmax(rationale_logits)
            # [batch, seq_len]
            rationale_logits = rationale_logits.view(batch_size, seq_length)

            # [batch, seq, hidden] [batch, seq_len, 1] = [batch, seq, hidden]
            final_hidden = sequence_output*rationale_logits.unsqueeze(2)
            sequence_output = final_hidden.view(batch_size*seq_length, hidden_size)

            logits = self.qa_outputs(sequence_output).view(batch_size, seq_length, 2)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)
            # [000,11111] 1代表了文章
            # [batch, seq_len] [batch, seq_len]
            rationale_logits = rationale_logits * attention_mask.float()
            # [batch, seq_len, 1] [batch, seq_len]
            start_logits = start_logits*rationale_logits
            end_logits = end_logits*rationale_logits

            if self.need_birnn:
                self.birnn.flatten_parameters()
                self.gru.flatten_parameters()
                sequence_output_IO, _ = self.birnn(sequence_output_IO)
                # sequence_output_IO, _ = self.gru(sequence_output_IO)
                
                
            sequence_output_IO = self.dropout(sequence_output_IO)
            emissions = self.hidden2tag(sequence_output_IO)

            # answers num
            switch_logits = self.qa_classifier(pooled_output)  # 用cls位置向量进行答案数量分类

            # extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
            # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            # sequence_output_sw = self.coref_layer(sequence_output_switch, extended_attention_mask)[0]
            #
            # switch_logits = self.qa_classifier(sequence_output_sw[:,0,:])

            # unk
            unk_logits = self.unk_ouputs(pooled_output)

            # doc_attn
            attention = self.doc_att(sequence_output)
            attention = attention.view(batch_size, seq_length)
            attention = attention*token_type_ids.float() + (1-token_type_ids.float())*VERY_NEGATIVE_NUMBER

            attention = F.softmax(attention, 1)
            attention = attention.unsqueeze(2)
            attention_pooled_output = attention*final_hidden
            attention_pooled_output = attention_pooled_output.sum(1)

            # 去掉attention
            # attention_pooled_output = pooled_output
            # yes_no_logits = self.yes_no_ouputs_noAttention(attention_pooled_output)

            yes_no_logits = self.yes_no_ouputs(attention_pooled_output)
            yes_logits, no_logits = yes_no_logits.split(1, dim=-1)

            # unk_yes_no_logits = self.ouputs_cls_3(attention_pooled_output)
            # unk_logits, yes_logits, no_logits = unk_yes_no_logits.split(1, dim=-1)

        else:
            # sequence_output = self.qa_dropout(sequence_output)
            logits = self.qa_outputs(sequence_output)
            # self attention
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            # answers num
            switch_logits = self.qa_classifier(pooled_output)  # 用cls位置向量进行答案数量分类

            # # unk yes_no_logits
            # pooled_output = self.unk_yes_no_outputs_dropout(pooled_output)
            unk_yes_no_logits = self.unk_yes_no_outputs(pooled_output)
            unk_logits, yes_logits, no_logits= unk_yes_no_logits.split(1, dim=-1)
        # # [batch, 1]
        # unk_logits = unk_logits.squeeze(-1)
        # yes_logits = yes_logits.squeeze(-1)
        # no_logits = no_logits.squeeze(-1)

        new_start_logits = torch.cat([start_logits, unk_logits, yes_logits, no_logits], 1)
        new_end_logits = torch.cat([end_logits, unk_logits, yes_logits, no_logits], 1)

        if self.answer_verification and start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            if len(answer_nums.size()) > 1:
                answer_nums = answer_nums.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = new_start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_losses = [(loss_fct(new_start_logits, _start_positions) * _span_mask) \
                            for (_start_positions, _span_mask) \
                            in zip(torch.unbind(start_positions, dim=1), torch.unbind(answer_masks, dim=1))]  # torch.unbind 移除指定维后，返回一个元组，包含了沿着指定维切片后的各个切片
            end_losses = [(loss_fct(new_end_logits, _end_positions) * _span_mask) \
                          for (_end_positions, _span_mask) \
                          in zip(torch.unbind(end_positions, dim=1), torch.unbind(answer_masks, dim=1))]
            
            loss_IO = -1 * self.crf(emissions, label_ids, mask=attention_mask.byte())

            switch_loss = loss_fct(switch_logits, answer_nums)

            # start_loss = loss_fct(new_start_logits, start_positions)
            # end_loss = loss_fct(new_end_logits, end_positions)

            rationale_positions = token_type_ids.float()
            alpha = 0.25
            gamma = 2.
            rationale_loss = -alpha * ((1 - rationale_logits) ** gamma) * rationale_positions * torch.log(
                rationale_logits + 1e-8) - (1 - alpha) * (rationale_logits ** gamma) * (
                                     1 - rationale_positions) * torch.log(1 - rationale_logits + 1e-8)
            rationale_loss = (rationale_loss*token_type_ids.float()).sum() / token_type_ids.float().sum()

            # s_e_loss = sum(start_losses + end_losses) + rationale_loss*self.beta
            # total_loss = torch.mean(s_e_loss + switch_loss)

            s_e_loss = sum(start_losses + end_losses)
            total_loss = torch.mean(s_e_loss + switch_loss + loss_IO) + rationale_loss * self.beta
            # total_loss = (start_losses + end_losses) / 2

            return total_loss

        elif start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = new_start_logits.size(1)
            start_positions.clamp_(1, ignored_index)
            end_positions.clamp_(1, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(new_start_logits, start_positions)
            end_loss = loss_fct(new_end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            IO_logits = self.crf.decode(emissions, attention_mask.byte())
            for io in IO_logits:
                while len(io) < 512:
                    io.append(0)
            IO_logits=torch.Tensor(IO_logits)
            IO_logits = IO_logits.cuda()
            return start_logits, end_logits, unk_logits, yes_logits, no_logits, switch_logits, IO_logits


class MultiLinearLayer(nn.Module):
    def __init__(self, layers, hidden_size, output_size, activation=None):
        super(MultiLinearLayer, self).__init__()
        self.net = nn.Sequential()

        for i in range(layers-1):
            self.net.add_module(str(i)+'linear', nn.Linear(hidden_size, hidden_size))
            self.net.add_module(str(i)+'relu', nn.ReLU(inplace=True))

        self.net.add_module('linear', nn.Linear(hidden_size, output_size))

    def forward(self, x):
        return self.net(x)

