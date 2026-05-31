"""
colab_runner.py - Pure Python Headless Execution Runner for Google Colab
"""
import sys
import os
import re
import subprocess
from pathlib import Path

# Add project root to sys.path to resolve imports cleanly
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from clipper_core import AutoClipperCore
from utils.colab_config import ColabConfig
from utils.ai_client_factory import create_ai_client

# Suppress window creations under Windows when simulated/tested locally
SUBPROCESS_FLAGS = 0
if sys.platform == "win32":
    SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW


def run_colab_pipeline(config_dict: dict) -> list:
    """Executes the video processing pipeline headlessly in Google Colab"""
    print("🚀 Initializing YT-Short-Clipper Colab Pipeline...")
    
    # 1. Config Loading
    colab_config = ColabConfig(config_dict)
    colab_config.validate()
    
    # 2. Setup Core Processor
    core = AutoClipperCore(
        client=None,
        output_dir=str(colab_config.output_dir),
        face_tracking_mode=colab_config.face_mode,
        subtitle_language=colab_config.subtitle_language,
        system_prompt=colab_config.system_prompt,
        temperature=colab_config.temperature,
        log_callback=lambda msg: print(msg),
        progress_callback=lambda status, p: print(f"📊 [Progress {p*100:.1f}%] {status}")
    )
    
    # 3. GPU Acceleration Detection
    try:
        import torch
        gpu_active = torch.cuda.is_available()
    except ImportError:
        gpu_active = False
        
    if gpu_active:
        print("⚡ GPU acceleration detected! Activating fast FFmpeg hardware codecs.")
        core.enable_gpu_acceleration(True)
    else:
        print("💻 Running on standard CPU. Encoding may take slightly longer.")
        core.enable_gpu_acceleration(False)
        
    # 4. Initialize Factory-Created Adapters
    provider = colab_config.config.get("provider", "openrouter")
    api_key = colab_config.config.get("api_key", "")
    
    primary_client = create_ai_client(provider, api_key)
    
    whisper_provider = colab_config.config.get("whisper_provider") or provider
    whisper_api_key = colab_config.config.get("whisper_api_key") or api_key
    
    caption_client = create_ai_client(whisper_provider, whisper_api_key)
    
    # Override client attributes on Core
    core.highlight_client = primary_client
    core.model = colab_config.config.get("model") or "auto"
    if core.model == "auto" and provider == "openrouter":
        core.model = "openrouter/free"
        
    core.caption_client = caption_client
    core.whisper_model = colab_config.config.get("whisper_model", "whisper-1")
    
    core.tts_client = primary_client
    core.tts_model = "tts-1"
    
    # yt-dlp preference
    from utils.helpers import is_ytdlp_module_available
    if is_ytdlp_module_available():
        core.ytdlp_path = "yt_dlp_module"
    else:
        core.ytdlp_path = "yt-dlp"

    # Step 1: Download Video & Subtitles
    print("\n🎬 [STEP 1] Downloading YouTube Video...")
    try:
        video_path, srt_path, video_info = core.download_video(colab_config.url)
        core.channel_name = video_info.get("channel", "") if video_info else ""
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return []
        
    # Step 2: Highlight extraction & parsing
    print("\n🔍 [STEP 2] Extracting Video Highlights...")
    highlights = []
    
    # Subtitle file missing or AI transcription forced
    if not srt_path or colab_config.subtitle_language == "none":
        print("  ⚠️ Subtitles unavailable or AI Transcribe requested. Transcribing audio track via Whisper API...")
        try:
            session_data = core.find_highlights_with_transcription(
                video_path=video_path,
                video_info=video_info,
                num_clips=colab_config.num_clips
            )
            highlights = session_data.get("highlights", [])
        except Exception as e:
            print(f"  ❌ Transcription/Highlight matching failed: {e}")
    else:
        try:
            transcript = core.parse_srt(srt_path)
            print("  ✓ Video transcript parsed successfully.")
            highlights = core.find_highlights(transcript, video_info, colab_config.num_clips)
        except Exception as e:
            print(f"  ⚠️ AI highlight detection failed: {e}")
            
    # Simple timestamp-based split fallback on failure
    if not highlights:
        print("  ⚠️ Fallback: Generating basic equal-length split highlights (90s duration).")
        probe_cmd = [core.ffmpeg_path, "-i", video_path, "-f", "null", "-"]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, creationflags=SUBPROCESS_FLAGS)
        duration_match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
        duration = 180.0
        if duration_match:
            h, m, s = duration_match.groups()
            duration = int(h) * 3600 + int(m) * 60 + float(s)
            
        segment_len = 90.0
        num_segments = min(colab_config.num_clips, int(duration // segment_len))
        if num_segments == 0:
            num_segments = 1
            segment_len = duration
            
        for i in range(num_segments):
            start_s = i * segment_len
            end_s = min(start_s + segment_len, duration)
            
            def form_time(seconds):
                hh = int(seconds // 3600)
                mm = int((seconds % 3600) // 60)
                ss = int(seconds % 60)
                ms = int((seconds % 1) * 1000)
                return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"
                
            highlights.append({
                "start_time": form_time(start_s),
                "end_time": form_time(end_s),
                "title": f"Highlight Segment {i+1}",
                "description": f"Auto-generated segment fallback {i+1}",
                "virality_score": 5,
                "hook_text": f"Bagian menarik ke-{i+1}"
            })
        print(f"  Generated {len(highlights)} backup clips.")
        
    # Step 3: Core processing (cut, crop, captions, hooks)
    print(f"\n⚙️ [STEP 3] Rendering {len(highlights)} Short Clips...")
    output_files = []
    total = len(highlights)
    
    for idx, highlight in enumerate(highlights, 1):
        print(f"\n🎞️ Rendering clip {idx}/{total}: {highlight['title']}")
        try:
            core.face_tracking_mode = colab_config.face_mode
            core.process_clip(
                video_path=video_path,
                highlight=highlight,
                index=idx,
                total_clips=total,
                add_captions=colab_config.add_captions,
                add_hook=colab_config.add_hook
            )
            
            # Identify output master file
            subdirs = sorted([d for d in colab_config.output_dir.iterdir() if d.is_dir() and d.name != "_temp"])
            if subdirs:
                latest_dir = subdirs[-1]
                master = latest_dir / "master.mp4"
                if master.exists():
                    output_files.append(str(master))
                    print(f"  ✓ Export completed at: {master}")
                else:
                    print(f"  ⚠️ Warning: Export completed but master.mp4 not found in {latest_dir}")
        except Exception as e:
            print(f"  ❌ Rendering failed for clip {idx}: {e}")
            
    # Core Temp files cleanup
    try:
        core.cleanup()
    except Exception:
        pass
        
    print("\n🎉 YT-Short-Clipper execution complete!")
    print(f"Total processed clips: {len(output_files)}")
    for filepath in output_files:
        print(f"  📁 {filepath}")
        
    return output_files


if __name__ == "__main__":
    # Test stub
    pass
