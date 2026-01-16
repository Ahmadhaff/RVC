import traceback
import logging

logger = logging.getLogger(__name__)

import numpy as np
import soundfile as sf
import torch
from io import BytesIO

from infer.lib.audio import load_audio, wav2
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.modules.vc.pipeline import Pipeline
from infer.modules.vc.utils import *


class VC:
    def __init__(self, config):
        self.n_spk = None
        self.tgt_sr = None
        self.net_g = None
        self.pipeline = None
        self.cpt = None
        self.version = None
        self.if_f0 = None
        self.version = None
        self.hubert_model = None

        self.config = config

    def get_vc(self, sid, *to_return_protect):
        logger.info("Get sid: " + sid)

        to_return_protect0 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[0] if self.if_f0 != 0 and to_return_protect else 0.5
            ),
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[1] if self.if_f0 != 0 and to_return_protect else 0.33
            ),
            "__type__": "update",
        }

        if sid == "" or sid == []:
            if (
                self.hubert_model is not None
            ):  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
                logger.info("Clean model cache")
                del (self.net_g, self.n_spk, self.hubert_model, self.tgt_sr)  # ,cpt
                self.hubert_model = self.net_g = self.n_spk = self.hubert_model = (
                    self.tgt_sr
                ) = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                ###楼下不这么折腾清理不干净
                self.if_f0 = self.cpt.get("f0", 1)
                self.version = self.cpt.get("version", "v1")
                if self.version == "v1":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs256NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs256NSFsid_nono(*self.cpt["config"])
                elif self.version == "v2":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs768NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs768NSFsid_nono(*self.cpt["config"])
                del self.net_g, self.cpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return (
                {"visible": False, "__type__": "update"},
                {
                    "visible": True,
                    "value": to_return_protect0,
                    "__type__": "update",
                },
                {
                    "visible": True,
                    "value": to_return_protect1,
                    "__type__": "update",
                },
                "",
                "",
            )
        person = f'{os.getenv("weight_root")}/{sid}'
        # Add .pth extension if not present and file doesn't exist
        if not os.path.exists(person) and os.path.exists(f"{person}.pth"):
            person = f"{person}.pth"
        logger.info(f"Loading: {person}")

        self.cpt = torch.load(person, map_location="cpu")
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }

        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*self.cpt["config"], is_half=self.config.is_half)

        del self.net_g.enc_q

        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)
        n_spk = self.cpt["config"][-3]
        index = {"value": get_index_path_from_model(sid), "__type__": "update"}
        logger.info("Select index: " + index["value"])

        return (
            (
                {"visible": True, "maximum": n_spk, "__type__": "update"},
                to_return_protect0,
                to_return_protect1,
                index,
                index,
            )
            if to_return_protect
            else {"visible": True, "maximum": n_spk, "__type__": "update"}
        )

    def vc_single(
        self,
        sid,
        input_audio_path,
        f0_up_key,
        f0_file,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    ):
        if input_audio_path is None:
            return "You need to upload an audio", None
        # Extract float value if protect is a dict (Gradio compatibility)
        if isinstance(protect, dict):
            protect = protect.get("value", 0.33) if "value" in protect else 0.33
        protect = float(protect)
        f0_up_key = int(f0_up_key)
        try:
            audio = load_audio(input_audio_path, 16000)
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            times = [0, 0, 0]

            if self.hubert_model is None:
                self.hubert_model = load_hubert(self.config)

            if file_index:
                file_index = (
                    file_index.strip(" ")
                    .strip('"')
                    .strip("\n")
                    .strip('"')
                    .strip(" ")
                    .replace("trained", "added")
                )
                # Convert to absolute path and resolve symlinks
                if file_index:
                    if not os.path.isabs(file_index):
                        # Try relative to current directory first
                        if os.path.exists(file_index) or os.path.islink(file_index):
                            file_index = os.path.abspath(file_index)
                        else:
                            # Try relative to logs directory
                            potential_path = os.path.join("logs", file_index)
                            if os.path.exists(potential_path) or os.path.islink(
                                potential_path
                            ):
                                file_index = os.path.abspath(potential_path)
                            else:
                                file_index = os.path.abspath(file_index)
                    else:
                        file_index = os.path.abspath(file_index)

                    # Resolve symlinks to get the actual file path
                    if os.path.islink(file_index) or (
                        os.path.exists(file_index) and os.path.islink(file_index)
                    ):
                        try:
                            resolved = os.path.realpath(file_index)
                            if os.path.exists(resolved):
                                file_index = resolved
                        except:
                            pass  # Keep original if resolution fails
            elif file_index2:
                file_index = file_index2
                if file_index:
                    if not os.path.isabs(file_index):
                        file_index = os.path.abspath(file_index)
                    # Resolve symlinks
                    if os.path.islink(file_index) or (
                        os.path.exists(file_index) and os.path.islink(file_index)
                    ):
                        try:
                            resolved = os.path.realpath(file_index)
                            if os.path.exists(resolved):
                                file_index = resolved
                        except:
                            pass
            else:
                file_index = ""  # 防止小白写错，自动帮他替换掉

            audio_opt = self.pipeline.pipeline(
                self.hubert_model,
                self.net_g,
                sid,
                audio,
                input_audio_path,
                times,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                self.if_f0,
                filter_radius,
                self.tgt_sr,
                resample_sr,
                rms_mix_rate,
                self.version,
                protect,
                f0_file,
            )
            if self.tgt_sr != resample_sr >= 16000:
                tgt_sr = resample_sr
            else:
                tgt_sr = self.tgt_sr
            # Check if index was actually used (check both symlink and resolved path)
            index_used = False
            if file_index:
                # Check if file exists (handles both regular files and symlinks)
                if os.path.exists(file_index) or os.path.islink(file_index):
                    # For symlinks, check if the target exists
                    if os.path.islink(file_index):
                        try:
                            resolved = os.path.realpath(file_index)
                            index_used = os.path.exists(resolved)
                        except:
                            index_used = os.path.exists(file_index)
                    else:
                        index_used = os.path.exists(file_index)

            index_info = "Index:\n%s." % file_index if index_used else "Index not used."
            return (
                "Success.\n%s\nTime:\nnpy: %.2fs, f0: %.2fs, infer: %.2fs."
                % (index_info, *times),
                (tgt_sr, audio_opt),
            )
        except:
            info = traceback.format_exc()
            logger.warning(info)
            return info, (None, None)

    def vc_multi(
        self,
        sid,
        dir_path,
        opt_root,
        paths,
        f0_up_key,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        format1,
    ):
        try:
            # Extract float value if protect is a dict (Gradio compatibility)
            if isinstance(protect, dict):
                protect = protect.get("value", 0.33) if "value" in protect else 0.33
            protect = float(protect)
            dir_path = (
                dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            )  # 防止小白拷路径头尾带了空格和"和回车
            opt_root = opt_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            os.makedirs(opt_root, exist_ok=True)
            try:
                if dir_path != "":
                    # Check if dir_path is a file or directory
                    if os.path.isfile(dir_path):
                        # If it's a file, treat it as a single file to convert
                        paths = [dir_path]
                    elif os.path.isdir(dir_path):
                        # If it's a directory, get all files in it
                        paths = [
                            os.path.join(dir_path, name)
                            for name in os.listdir(dir_path)
                        ]
                    else:
                        # Path doesn't exist
                        yield f"Error: Path '{dir_path}' does not exist."
                        return
                else:
                    paths = [path.name for path in paths]
            except Exception as e:
                logger.warning(f"Error processing paths: {str(e)}")
                traceback.print_exc()
                # Fallback to uploaded files if available
                try:
                    paths = [path.name for path in paths]
                except:
                    yield f"Error: Could not process input paths. {str(e)}"
                    return
            infos = []
            for path in paths:
                info, opt = self.vc_single(
                    sid,
                    path,
                    f0_up_key,
                    None,
                    f0_method,
                    file_index,
                    file_index2,
                    # file_big_npy,
                    index_rate,
                    filter_radius,
                    resample_sr,
                    rms_mix_rate,
                    protect,
                )
                if "Success" in info:
                    try:
                        tgt_sr, audio_opt = opt
                        if format1 in ["wav", "flac"]:
                            sf.write(
                                "%s/%s.%s"
                                % (opt_root, os.path.basename(path), format1),
                                audio_opt,
                                tgt_sr,
                            )
                        else:
                            path = "%s/%s.%s" % (
                                opt_root,
                                os.path.basename(path),
                                format1,
                            )
                            with BytesIO() as wavf:
                                sf.write(wavf, audio_opt, tgt_sr, format="wav")
                                wavf.seek(0, 0)
                                with open(path, "wb") as outf:
                                    wav2(wavf, outf, format1)
                    except:
                        info += traceback.format_exc()
                infos.append("%s->%s" % (os.path.basename(path), info))
                yield "\n".join(infos)
            yield "\n".join(infos)
        except GeneratorExit:
            # Re-raise GeneratorExit to allow proper generator cleanup
            raise
        except Exception as e:
            yield traceback.format_exc()
