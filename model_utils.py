from transformers import AutoTokenizer, EsmForTokenClassification

def load_tokenizer_and_model(model_name="facebook/esm2_t6_8M_UR50D", num_labels=2):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model

def freeze_layers(model, freeze_up_to=4):
    for name, param in model.esm.named_parameters():
        if any(f"layer.{i}." in name for i in range(freeze_up_to)):
            param.requires_grad = False
    return model
