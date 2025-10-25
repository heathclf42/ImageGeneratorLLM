# Comprehensive AI Pipeline Flowchart Design

## Layout Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT STAGE (All Types)                       │
│  [Text Input] [Image Input] [Audio Input] [Control Input]       │
└────────┬────────────────┬────────────────┬───────────────────────┘
         │                │                │
         ▼                ▼                ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│  Text Branch   │ │ Image Branch   │ │ Audio Branch   │
├────────────────┤ ├────────────────┤ ├────────────────┤
│ • Tokenization │ │ • Image Encode │ │ • Audio Encode │
│ • Embedding    │ │ • VAE Encode   │ │ • Feature Ext  │
│ • Attention    │ │ • Latent Space │ │ • Spectrogram  │
└───────┬────────┘ └───────┬────────┘ └───────┬────────┘
        │                  │                  │
        └──────────┬───────┴──────────────────┘
                   ▼
        ┌──────────────────────┐
        │   PROCESSING CORE     │
        │                       │
        │  • Model Loading      │
        │  • Inference          │
        │  • Transformation     │
        └───────────┬───────────┘
                    │
        ┌───────────┴────────────────────┐
        │                                │
        ▼                                ▼
┌────────────────┐          ┌────────────────┐
│ Media Decode   │          │  Text Decode   │
├────────────────┤          ├────────────────┤
│ • VAE Decode   │          │ • Detokenize   │
│ • Vocoder      │          │ • Format       │
│ • Frame Render │          │ • Output       │
└────────┬───────┘          └────────┬───────┘
         │                           │
         └───────────┬───────────────┘
                     ▼
          ┌──────────────────┐
          │  OUTPUT STAGE     │
          │  [Result Display] │
          └──────────────────┘
```

## Component Mapping by Mode

### text2img:
Input → Tokenize → Embed → Diffusion → VAE Decode → Output

### img2img:
Image Input → VAE Encode → Diffusion → VAE Decode → Output

### llm:
Text Input → Tokenize → Attention Layers → Detokenize → Text Output

### text2audio:
Text Input → Phonemes → TTS Synthesis → Vocoder → Audio Output

### audio2text:
Audio Input → Features → ASR → Detokenize → Text Output

### text2video:
Text Input → Embed → Video Diffusion → Frame Render → Video Output

### img2video:
Image Input → VAE Encode → Temporal Diffusion → Frame Render → Video Output

### controlnet:
Text + Control → Embed + Structure → Guided Diffusion → VAE Decode → Output
