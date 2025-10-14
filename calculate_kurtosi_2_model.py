from scipy.stats import kurtosis
from transformers import AutoModel

def calc_ku(model, layer_kind=None):
    """Calcola la kurtosi aggregata di un modello."""
    model_ku = 0
    for name, layer in model.state_dict().items():
        # Consideriamo solo matrici quadrate (pesi di layer lineari)
        if len(layer.shape) != 2 or layer.shape[0] != layer.shape[1]:
            continue

        if layer_kind is not None and layer_kind not in name:
            continue

        ku = kurtosis(layer.flatten().detach().cpu().numpy(), fisher=False)
        model_ku += ku
    return model_ku


# Nomi dei modelli su Hugging Face
parent_model_name = "MoTHer-VTHR/VTHR-FT-ModelTree_1-Depth_1-Node_eCWU4rGY"
child_model_name = "MoTHer-VTHR/VTHR-FT-ModelTree_1-Depth_2-Node_gpnAyv4k"

# Caricamento diretto da Hugging Face
model_a = AutoModel.from_pretrained(parent_model_name)
model_b = AutoModel.from_pretrained(child_model_name)

# Calcolo kurtosi (puoi cambiare layer_kind: 'output.dense', 'attention.query', ecc.)
ku_a = calc_ku(model_a, layer_kind="output.dense")
ku_b = calc_ku(model_b, layer_kind="output.dense")

print("kurtosi A (BERT base):", ku_a)
print("kurtosi B (BERT banking):", ku_b)
print("A > B ? ->", ku_a > ku_b)
