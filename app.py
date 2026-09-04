import streamlit as st
import os
import torch
import static_ffmpeg
import tempfile
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from peft import PeftModel

# FFmpeg path for Windows
static_ffmpeg.add_paths()

CHECKPOINT_PATH = r"C:\Users\tisaj\Downloads\nepali_whisper_final_model\checkpoint-400"
MODEL_ID = "openai/whisper-large-v3-turbo"

NEPALI_DICT = {
    "नेपालि": "नेपाली", "तपाँई": "तपाईं", "हजुरबा": "हजुरबुवा", "तिमि": "तिमी",
    "तिमिको": "तिमीको", "तिमीलाइ": "तिमीलाई", "तिमिले": "तिमीले", "उहा": "उहाँ",
    "उहाको": "उहाँको", "उहालाई": "उहाँलाई", "उहाले": "उहाँले", "हामि": "हामी",
    "हामिको": "हामीको", "हामिलाई": "हामीलाई", "हामिले": "हामीले", "उनि": "उनी",
    "उनिको": "उनीको", "उनिलाई": "उनीलाई", "उनिले": "उनीले", "तिनि": "तिनी",
    "तिनिको": "तिनीको", "तिनिलाई": "तिनीलाई", "तिनिले": "तिनीले", "गळत": "गलत",
    "गळो": "गलत", "गलतट": "गलत", "गलतछ": "गलत", "काठमाडौ": "काठमाडौँ",
    "काठमाण्डौ": "काठमाडौँ", "काठमाण्डु": "काठमाडौँ", "पोखरामा": "पोखरामा",
    "ललितपुर": "ललितपुर", "भक्तपुर": "भक्तपुर", "राम्रोसग": "राम्रोसँग",
    "हाम्रोसग": "हाम्रोसँग", "तिम्रोसग": "तिम्रोसँग", "तपाईसग": "तपाईंसँग",
    "उहासग": "उहाँसँग", "उनीसग": "उनीसँग", "यससग": "यससँग", "त्यससग": "त्यससँग",
    "गर्नुहोस": "गर्नुहोस्", "आउनुहोस": "आउनुहोस्", "जानुहोस": "जानुहोस्",
    "खानुहोस": "खानुहोस्", "बस्नुहोस": "बस्नुहोस्", "उठ्नुहोस": "उठ्नुहोस्",
    "हेर्नुहोस": "हेर्नुहोस्", "सुन्नुहोस": "सुन्नुहोस्", "भन्नुहोस": "भन्नुहोस्",
    "लेख्नुहोस": "लेख्नुहोस्", "पढ्नुहोस": "पढ्नुहोस्", "सिक्नुहोस": "सिक्नुहोस्",
    "बोल्नुहोस": "बोल्नुहोस्", "हाँस्नुहोस": "हाँस्नुहोस्", "रुनुहोस": "रुने गर्नुहोस्",
    "धोइदिनुहोस": "धोइदिनुहोस्", "नेपालिह": "नेपाली", "काठमाडौं": "काठमाडौँ",
    "धन्यबाद": "धन्यवाद", "नमस्ते": "नमस्ते", "पर्‍यो": "पर्यो",
    "कुराहरु": "कुराहरू", "साथीहरु": "साथीहरू", "मान्छेहरु": "मान्छेहरू",
}

def apply_corrections(text):
    for wrong, right in NEPALI_DICT.items():
        text = text.replace(wrong, right)
    return text


@st.cache_resource
def load_whisper_model():
    st.info("Loading model into RAM")
    
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="nepali", task="transcribe")
    
    base_model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    )
    
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    model = model.merge_and_unload()
    model.eval()
    
    asr = pipeline(
        "automatic-speech-recognition", 
        model=model, 
        tokenizer=processor.tokenizer, 
        feature_extractor=processor.feature_extractor
    )
    return asr

# streamlit
st.set_page_config(page_title="Nepali Whisper", page_icon="🎙️")
st.title("Nepali Speech To Text")
st.write("Upload an audio file to transcribe.")

asr_pipeline = load_whisper_model()

uploaded_file = st.file_uploader("Choose an audio file (WAV, MP3, M4A)", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Transcribe Now"):
        with st.spinner("Processing Nepali Audio on CPU..."):
            try:
                result = asr_pipeline(tmp_path, generate_kwargs={"language": "nepali", "task": "transcribe"})
                
                final_text = apply_corrections(result["text"])
                
                st.subheader("Results:")
                st.success(final_text)
                
                st.text_area("To copy:", value=final_text, height=150)
                
            except Exception as e:
                st.error(f"Error during transcription: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)




# import streamlit as st
# import os
# import torch
# import static_ffmpeg
# import tempfile
# import numpy as np
# from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
# from peft import PeftModel

# # Add FFmpeg to path
# static_ffmpeg.add_paths()

# # ==========================================
# # 1. SETTINGS
# # ==========================================
# CHECKPOINT_PATH = r"C:\Users\tisaj\Downloads\nepali_whisper_final_model\checkpoint-400"
# MODEL_ID = "openai/whisper-large-v3-turbo"

# NEPALI_DICT = {
#     "नेपालि": "नेपाली", "तपाँई": "तपाईं", "हजुरबा": "हजुरबुवा", "तिमि": "तिमी",
#     "तिमिको": "तिमीको", "तिमीलाइ": "तिमीलाई", "तिमिले": "तिमीले", "उहा": "उहाँ",
#     "उहाको": "उहाँको", "उहालाई": "उहाँलाई", "उहाले": "उहाँले", "हामि": "हामी",
#     "हामिको": "हामीको", "हामिलाई": "हामीलाई", "हामिले": "हामीले", "उनि": "उनी",
#     "उनिको": "उनीको", "उनिलाई": "उनीलाई", "उनिले": "उनीले", "तिनि": "तिनी",
#     "तिनिको": "तिनीको", "तिनिलाई": "तिनीलाई", "तिनिले": "तिनीले", "गळत": "गलत",
#     "गळो": "गलत", "गलतट": "गलत", "गलतछ": "गलत", "काठमाडौ": "काठमाडौँ",
#     "काठमाण्डौ": "काठमाडौँ", "काठमाण्डु": "काठमाडौँ", "पोखरामा": "पोखरामा",
#     "ललितपुर": "ललितपुर", "भक्तपुर": "भक्तपुर", "राम्रोसग": "राम्रोसँग",
#     "हाम्रोसग": "हाम्रोसँग", "तिम्रोसग": "तिम्रोसँग", "तपाईसग": "तपाईंसँग",
#     "उहासग": "उहाँसँग", "उनीसग": "उनीसँग", "यससग": "यससँग", "त्यससग": "त्यससँग",
#     "गर्नुहोस": "गर्नुहोस्", "आउनुहोस": "आउनुहोस्", "जानुहोस": "जानुहोस्",
#     "खानुहोस": "खानुहोस्", "बस्नुहोस": "बस्नुहोस्", "उठ्नुहोस": "उठ्नुहोस्",
#     "हेर्नुहोस": "हेर्नुहोस्", "सुन्नुहोस": "सुन्नुहोस्", "भन्नुहोस": "भन्नुहोस्",
#     "लेख्नुहोस": "लेख्नुहोस्", "पढ्नुहोस": "पढ्नुहोस्", "सिक्नुहोस": "सिक्नुहोस्",
#     "बोल्नुहोस": "बोल्नुहोस्", "हाँस्नुहोस": "हाँस्नुहोस्", "रुनुहोस": "रुने गर्नुहोस्",
#     "धोइदिनुहोस": "धोइदिनुहोस्", "नेपालिह": "नेपाली", "काठमाडौं": "काठमाडौँ",
#     "धन्यबाद": "धन्यवाद", "नमस्ते": "नमस्ते", "पर्‍यो": "पर्यो",
#     "कुराहरु": "कुराहरू", "साथीहरु": "साथीहरू", "मान्छेहरु": "मान्छेहरू",
# }

# def apply_corrections(text):
#     for wrong, right in NEPALI_DICT.items():
#         text = text.replace(wrong, right)
#     return text

# # ==========================================
# # 2. CACHED MODEL LOADING (CPU SAFE)
# # ==========================================
# @st.cache_resource
# def load_whisper_model():
#     # Use standard print so we can see it in the terminal
#     print("--- Loading Model into RAM (CPU-Safe Mode) ---")
    
#     # 1. Load Processor
#     processor = WhisperProcessor.from_pretrained(MODEL_ID, language="nepali", task="transcribe")
    
#     # 2. Load Base Model (FORCE FLOAT32 FOR CPU)
#     base_model = WhisperForConditionalGeneration.from_pretrained(
#         MODEL_ID, 
#         torch_dtype=torch.float32,  # <--- Changed from float16
#         low_cpu_mem_usage=True,
#         device_map="cpu"            # <--- Explicitly tell it to stay on CPU
#     )
    
#     # 3. Attach LoRA weights
#     model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
#     model = model.merge_and_unload()
#     model.eval()
    
#     # 4. Create Pipeline
#     asr = pipeline(
#         "automatic-speech-recognition", 
#         model=model, 
#         tokenizer=processor.tokenizer, 
#         feature_extractor=processor.feature_extractor,
#         device=-1 # -1 means CPU
#     )
#     print("--- Model is ready and cached! ---")
#     return asr

# # ==========================================
# # 3. STREAMLIT UI
# # ==========================================
# st.set_page_config(page_title="Nepali Whisper ASR", page_icon="🎙️")
# st.title("🇳🇵 Trained Nepali Whisper ASR")

# # Load model
# try:
#     asr_pipeline = load_whisper_model()
#     st.success("Model loaded successfully!")
# except Exception as e:
#     st.error(f"Failed to load model: {e}")
#     st.stop()

# uploaded_file = st.file_uploader("Upload Audio (WAV, MP3, M4A)", type=["wav", "mp3", "m4a"])

# if uploaded_file is not None:
#     # Create temp file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
#         tmp.write(uploaded_file.getvalue())
#         tmp_path = tmp.name

#     st.audio(uploaded_file)
    
#     if st.button("Transcribe Now"):
#         with st.spinner("Processing... Please wait (30-60 seconds)"):
#             try:
#                 # Transcribe
#                 result = asr_pipeline(tmp_path, generate_kwargs={"language": "nepali", "task": "transcribe"})
#                 final_text = apply_corrections(result["text"])
                
#                 st.subheader("Transcription:")
#                 st.write(final_text)
#                 st.text_area("Copy Result:", value=final_text)
#             except Exception as e:
#                 st.error(f"Transcription Error: {e}")
#             finally:
#                 if os.path.exists(tmp_path):
#                     os.remove(tmp_path)