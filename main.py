from data_utils import parse_fasta, parse_label_fasta
from dataset import ProteinDataset, DataCollatorForTokenClassification
from model_utils import load_tokenizer_and_model, freeze_layers
from train import get_training_args, run_trainer
from evaluate import evaluate_model

def main():
    # Load data
    train_seq = parse_fasta('data/ZN/ZN_train_set.fasta')
    train_lbl = parse_label_fasta('data/ZN/ZN_train_labels.fasta')
    eval_seq = parse_fasta('data/ZN/ZN_eval_set.fasta')
    eval_lbl = parse_label_fasta('data/ZN/ZN_eval_labels.fasta')
    test_seq = parse_fasta('data/ZN/ZN_test_set.fasta')
    test_lbl = parse_label_fasta('data/ZN/ZN_test_labels.fasta')

    # Tokenizer & datasets
    tokenizer, model = load_tokenizer_and_model()
    model = freeze_layers(model)
    train_ds = ProteinDataset(train_seq, train_lbl, tokenizer)
    eval_ds = ProteinDataset(eval_seq, eval_lbl, tokenizer)
    test_ds = ProteinDataset(test_seq, test_lbl, tokenizer)
    collator = DataCollatorForTokenClassification(tokenizer)

    # Train
    args = get_training_args()
    trainer = run_trainer(model, tokenizer, train_ds, eval_ds, collator, args)

    # Evaluate
    evaluate_model(trainer, test_ds)

if __name__ == "__main__":
    main()
