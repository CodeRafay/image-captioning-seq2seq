"""
Neural Storyteller — Image Captioning with Seq2Seq
Streamlit App for generating captions from images using a pre-trained
ResNet50 Encoder + LSTM Decoder model.
"""

import io
import pickle

import requests
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ─────────────────────────────────────────────
# 1. Page config & custom CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Neural Storyteller — Image Captioning",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Gradient header bar ── */
.header-bar {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
    padding: 2rem 2.5rem;
    border-radius: 1rem;
    margin-bottom: 1.5rem;
    color: white;
    text-align: center;
}
.header-bar h1 {
    margin: 0;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.header-bar p {
    margin: 0.4rem 0 0;
    font-size: 1rem;
    opacity: 0.85;
}

/* ── Caption card ── */
.caption-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 1rem;
    padding: 1.8rem 2rem;
    margin-top: 1.2rem;
    color: #e0e7ff;
    font-size: 1.25rem;
    line-height: 1.6;
    text-align: center;
    animation: fadeSlideIn 0.6s ease-out;
}
.caption-card .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #a5b4fc;
    margin-bottom: 0.5rem;
}
.caption-card .text {
    font-weight: 600;
    font-size: 1.35rem;
}

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Sidebar styling ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0a2e 0%, #1a1145 100%);
}
section[data-testid="stSidebar"] * {
    color: #c7d2fe !important;
}
section[data-testid="stSidebar"] .stRadio > label,
section[data-testid="stSidebar"] .stSelectbox > label {
    font-weight: 600;
}

/* ── Image container ── */
.image-container {
    border-radius: 1rem;
    overflow: hidden;
    border: 2px solid rgba(139, 92, 246, 0.25);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
}
.image-container img {
    width: 100%;
    display: block;
}

/* ── Info box ── */
.info-box {
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 0.75rem;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    font-size: 0.88rem;
    color: #a5b4fc;
    line-height: 1.5;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 2. Model definitions (same as training notebook)
# ─────────────────────────────────────────────
class Encoder(nn.Module):
    """Projects 2048-dim ResNet50 feature vector → hidden_size."""

    def __init__(self, feature_size=2048, hidden_size=512):
        super().__init__()
        self.fc = nn.Linear(feature_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, features):
        return self.dropout(self.relu(self.fc(features)))


class Decoder(nn.Module):
    """LSTM decoder that generates captions word-by-word."""

    def __init__(self, embed_size=256, hidden_size=512,
                 vocab_size=7727, num_layers=1, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        lstm_dropout = dropout if num_layers > 1 else 0
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True, dropout=lstm_dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, captions, hidden, cell):
        embeddings = self.dropout(self.embed(captions))
        outputs, (hidden, cell) = self.lstm(embeddings, (hidden, cell))
        return self.fc(outputs), hidden, cell

    def init_hidden_state(self, encoder_out):
        hidden = encoder_out.unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell = torch.zeros_like(hidden)
        return hidden, cell


# ─────────────────────────────────────────────
# 3. Loading helpers (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading vocabulary…")
def load_vocab(path: str = "flickr30k_vocab.pkl"):
    with open(path, "rb") as f:
        vocab = pickle.load(f)
    return vocab


@st.cache_resource(show_spinner="Loading captioning model…")
def load_caption_model(path: str = "caption_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    embed_size = checkpoint["embed_size"]
    hidden_size = checkpoint["hidden_size"]
    vocab_size = checkpoint["vocab_size"]
    num_layers = checkpoint["num_layers"]

    encoder = Encoder(feature_size=2048, hidden_size=hidden_size).to(device)
    decoder = Decoder(embed_size=embed_size, hidden_size=hidden_size,
                      vocab_size=vocab_size, num_layers=num_layers).to(device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    encoder.eval()
    decoder.eval()
    return encoder, decoder, device


@st.cache_resource(show_spinner="Loading ResNet50 feature extractor…")
def load_resnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Remove final classification layer → outputs (batch, 2048, 1, 1)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).to(device)
    feature_extractor.eval()
    return feature_extractor, device


# Image transform matching training pipeline
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])


# ─────────────────────────────────────────────
# 4. Inference functions
# ─────────────────────────────────────────────
def extract_features(image: Image.Image, feature_extractor, device):
    """Extract 2048-dim feature vector from a PIL image."""
    img_tensor = IMG_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = feature_extractor(img_tensor).view(1, -1)  # (1, 2048)
    return features


def greedy_search(features, encoder, decoder, device, vocab, max_len=80):
    """Generate caption via greedy decoding."""
    word2idx = vocab["word2idx"]
    idx2word = vocab["idx2word"]

    with torch.no_grad():
        encoder_out = encoder(features)
        hidden, cell = decoder.init_hidden_state(encoder_out)

        caption_ids = [word2idx["<start>"]]
        for _ in range(max_len):
            inp = torch.tensor([[caption_ids[-1]]], dtype=torch.long, device=device)
            logits, hidden, cell = decoder(inp, hidden, cell)
            next_word = logits.argmax(2).item()
            if next_word == word2idx["<end>"]:
                break
            caption_ids.append(next_word)

    return " ".join(idx2word.get(i, "<unk>") for i in caption_ids[1:])


def beam_search(features, encoder, decoder, device, vocab,
                beam_width=5, max_len=80):
    """Generate caption via beam search decoding."""
    word2idx = vocab["word2idx"]
    idx2word = vocab["idx2word"]

    with torch.no_grad():
        encoder_out = encoder(features)
        hidden, cell = decoder.init_hidden_state(encoder_out)

        # (sequence, log_prob, hidden, cell)
        beams = [([word2idx["<start>"]], 0.0, hidden, cell)]
        completed = []

        for _ in range(max_len):
            candidates = []
            for seq, score, h, c in beams:
                if seq[-1] == word2idx["<end>"]:
                    completed.append((seq, score, h, c))
                    continue
                inp = torch.tensor([[seq[-1]]], dtype=torch.long, device=device)
                logits, h_new, c_new = decoder(inp, h, c)
                probs = torch.log_softmax(logits.squeeze(1), dim=-1)
                topk_probs, topk_idx = probs.topk(beam_width, dim=-1)
                topk_probs = topk_probs.squeeze(0)
                topk_idx = topk_idx.squeeze(0)
                for k in range(beam_width):
                    nw = topk_idx[k].item()
                    ns = score + topk_probs[k].item()
                    candidates.append((seq + [nw], ns, h_new, c_new))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
            beams = candidates[:beam_width]

            if all(b[0][-1] == word2idx["<end>"] for b in beams):
                completed.extend(beams)
                break

        completed.extend(beams)
        completed.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
        best = completed[0][0]

    words = []
    for idx in best:
        w = idx2word.get(idx, "<unk>")
        if w in ("<start>", "<end>"):
            continue
        words.append(w)
    return " ".join(words)


# ─────────────────────────────────────────────
# 5. Streamlit UI
# ─────────────────────────────────────────────
def main():
    # ── Header ──
    st.markdown(
        '<div class="header-bar">'
        "<h1>🖼️ Neural Storyteller</h1>"
        "<p>Image Captioning powered by ResNet50 &amp; LSTM Seq2Seq</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        search_method = st.radio(
            "Decoding strategy",
            ["Greedy Search", "Beam Search"],
            index=1,
            help="Greedy always picks the top word. Beam Search explores multiple paths for better captions.",
        )

        beam_width = 5
        if search_method == "Beam Search":
            beam_width = st.slider("Beam width", min_value=2, max_value=15,
                                   value=5, step=1,
                                   help="Higher = more exploration, slower.")

        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown(
            "This app uses a **ResNet50** encoder and an **LSTM** decoder "
            "trained on the [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) dataset.\n\n"
            "[GitHub Repo](https://github.com/CodeRafay/image-captioning-seq2seq)"
        )

        device_label = "🟢 GPU" if torch.cuda.is_available() else "🔵 CPU"
        st.markdown(f"**Device:** {device_label}")

    # ── Load models (cached) ──
    vocab = load_vocab()
    encoder, decoder, device = load_caption_model()
    feature_extractor, _ = load_resnet()

    # ── Image input ──
    st.markdown("### 📸 Provide an Image")

    tab_upload, tab_camera, tab_url = st.tabs([
        "📁  Upload File", "📷  Camera", "🔗  Paste URL"
    ])

    image: Image.Image | None = None

    with tab_upload:
        uploaded = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")

    with tab_camera:
        cam_photo = st.camera_input("Take a photo")
        if cam_photo is not None:
            image = Image.open(cam_photo).convert("RGB")

    with tab_url:
        url = st.text_input("Paste an image URL",
                            placeholder="https://example.com/photo.jpg")
        if url:
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                image = Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception as exc:
                st.error(f"Could not fetch image: {exc}")

    # ── Generate caption ──
    if image is not None:
        col_img, col_caption = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_caption:
            with st.spinner("Generating caption…"):
                features = extract_features(image, feature_extractor, device)

                if search_method == "Greedy Search":
                    caption = greedy_search(features, encoder, decoder,
                                           device, vocab)
                else:
                    caption = beam_search(features, encoder, decoder,
                                         device, vocab,
                                         beam_width=beam_width)

            method_label = search_method
            if search_method == "Beam Search":
                method_label += f" (width {beam_width})"

            st.markdown(
                f'<div class="caption-card">'
                f'<div class="label">{method_label}</div>'
                f'<div class="text">"{caption}"</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="info-box">'
                "💡 <b>Tip:</b> Try switching between <em>Greedy</em> and "
                "<em>Beam Search</em> in the sidebar to compare results. "
                "Higher beam widths explore more possibilities but take longer."
                "</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("👆 Upload, capture, or paste a URL above to get started!")


if __name__ == "__main__":
    main()
