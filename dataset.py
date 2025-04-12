import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_id = list(self.sequences.keys())[idx]
        sequence = self.sequences[seq_id]
        label = self.labels[seq_id]
        inputs = self.tokenizer(sequence, add_special_tokens=False, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = torch.tensor(label, dtype=torch.long)
        return inputs

class DataCollatorForTokenClassification:
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        attention_masks = [torch.ones_like(iid) for iid in input_ids]

        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

        if self.max_length:
            input_ids_padded = input_ids_padded[:, :self.max_length]
            attention_padded = attention_padded[:, :self.max_length]
            labels_padded = labels_padded[:, :self.max_length]

        return {
            'input_ids': input_ids_padded,
            'attention_mask': attention_padded,
            'labels': labels_padded
        }
