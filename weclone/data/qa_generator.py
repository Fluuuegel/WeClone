import json
import os
import re
import subprocess  # nosec
import sys
from typing import List, Union, cast

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pandas as pd
from pandas import Timestamp

from weclone.core.PII.pii_detector import ChinesePIIDetector, PIIDetector
from weclone.data.chat_parsers.telegram_parser import process_telegram_dataset
from weclone.data.clean.strategies import LLMCleaningStrategy, OlineLLMCleaningStrategy
from weclone.data.models import (
    ChatMessage,
    CutMessage,
    Message,
    QaPair,
    cut_type_list,
    skip_type_list,
)
from weclone.data.qq_emojis import QQ_EMOJI_CODES
from weclone.data.strategies import TimeWindowStrategy
from weclone.data.utils import ImageToTextProcessor, check_image_file_exists
from weclone.utils.config import load_config
from weclone.utils.config_models import DataModality, LanguageType, PlatformType, WCMakeDatasetConfig
from weclone.utils.log import logger


class DataProcessor:
    def __init__(self):
        self.config = cast(WCMakeDatasetConfig, load_config(arg_type="make_dataset"))
        self.csv_folder = "./dataset/csv"
        self.system_prompt = self.config.default_system
        self.enable_clean = self.config.clean_dataset.enable_clean

        # message type
        self.QaPair = QaPair

        self.include_type = self.config.include_type
        if self.config.platform == PlatformType.CHAT:
            self.cut_type_list = cut_type_list.get_items(lang="zh_CN")
            self.skip_type_list = skip_type_list.get_items(lang="zh_CN")
            self.include_type = cut_type_list.translate_batch(
                texts=[t for t in self.include_type if t.lower() != "text"]
            )
            self.cut_type_list = [t for t in self.cut_type_list if t not in self.include_type]
        elif self.config.platform == PlatformType.TELEGRAM:
            self.cut_type_list = cut_type_list.get_items(lang="en")
            self.skip_type_list = skip_type_list.get_items(lang="en")
            self.include_type = [t for t in self.include_type if t.lower() != "text"]
            self.cut_type_list = [t for t in self.cut_type_list if t not in self.include_type]
            if DataModality.STICKER in self.include_type:
                self.skip_type_list.remove("sticker")

        # blocked words
        config_blocked_words = self.config.blocked_words
        file_blocked_words = []
        try:
            with open("./dataset/blocked_words.json", encoding="utf-8") as f:
                file_blocked_words = json.load(f).get("blocked_words", [])
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        self.blocked_words = list(set(config_blocked_words + file_blocked_words))
        # logger.info(f"Chat record blocked words: {self.blocked_words}")

        # combine strategy
        if self.config.single_combine_strategy == "time_window":
            self.single_combine_strategy = TimeWindowStrategy(
                time_window=self.config.single_combine_time_window * 60,
                is_single_chat=True,
            )

        if self.config.qa_match_strategy == "time_window":
            self.qa_match_strategy = TimeWindowStrategy(
                time_window=self.config.qa_match_time_window * 60,
                is_single_chat=False,
            )

        # PII detection
        if self.config.language == LanguageType.ZH:
            self.pii_detector = ChinesePIIDetector()
        else:
            self.pii_detector = PIIDetector(language=self.config.language)

        # dataset cleaning
        clean_dataset_config = self.config.clean_dataset

        if self.enable_clean:
            if clean_dataset_config.clean_strategy == "llm":
                if self.config.online_llm_clear:
                    self.clean_strategy = OlineLLMCleaningStrategy(make_dataset_config=self.config)
                else:
                    from llamafactory.extras.packages import is_vllm_available

                    if not is_vllm_available():
                        logger.error("vLLM is not available, dataset cleaning is not supported.")
                        sys.exit(1)
                    else:
                        self.clean_strategy = LLMCleaningStrategy(make_dataset_config=self.config)

        vision_config = self.config.vision_api
        if vision_config.enable and vision_config.api_key:
            self.image_processor = ImageToTextProcessor(
                api_url=vision_config.api_url,  # type: ignore
                api_key=vision_config.api_key,  # type: ignore
                model_name=vision_config.model_name,  # type: ignore
                config=self.config,
            )
            logger.info(f"ImageToText functionality enabled, model: {self.image_processor.model_name}")
        else:
            self.image_processor = None

        self.c = self.config

        self.relations = {}
        self.name_map: dict = {}
        self._mention_re = None  # built by _build_nickname_map() before processing starts

    def main(self):
        self.pre_parse_chat_dataset()

        if not os.path.exists(self.csv_folder) or not os.listdir(self.csv_folder):
            logger.error(
                f"Error: Directory '{self.csv_folder}' does not exist or is empty. Please check the path and ensure it contains CSV chat data files."
            )
            sys.exit(1)

        csv_files = self.get_csv_files()
        self._build_nickname_map(csv_files)
        logger.info(f"Found {len(csv_files)} CSV files in total, starting processing, please be patient...")
        message_list: List[ChatMessage] = []
        for csv_file in csv_files:
            logger.debug(f"Starting to process CSV file: {csv_file}")
            chat_messages = self.load_file(csv_file)
            message_list.extend(self.group_consecutive_messages(messages=chat_messages))
            # self.process_by_msgtype(chat_message)
            logger.debug(f"Processing completed: {csv_file}, loaded {len(chat_messages)} messages in total")
        qa_res = self.match_qa(messages=message_list)
        qa_res = [item for item in qa_res if isinstance(item, QaPair)]

        if self.image_processor:
            logger.info("Starting image recognition process...")
            qa_res = self.image_processor._process_images_in_parallel(qa_res)
            logger.info("Image recognition process completed.")

        if self.enable_clean:
            self.clean_strategy.judge(qa_res)  # type: ignore

        self.save_result(qa_res)
        self._execute_length_cdf_script()

        logger.success(
            f"Chat record processing successful, obtained {len(qa_res)} data entries in total, saved to ./dataset/res_csv/sft/sft-my.json"
        )

    def pre_parse_chat_dataset(self):
        if self.c.platform == PlatformType.TELEGRAM:
            process_telegram_dataset(self.config)

    def _execute_length_cdf_script(self):
        """Execute the length_cdf.py script to calculate cutoff_len."""
        try:
            python_executable = sys.executable
            script_path = os.path.join("weclone", "utils", "length_cdf.py")

            command_parts = [
                python_executable,
                script_path,
                f'--model_name_or_path="{self.c.model_name_or_path}"',
                f'--dataset="{self.c.dataset}"',
                f'--dataset_dir="{self.c.dataset_dir}"',
                f'--template="{self.c.template}"',
                "--interval=512",
            ]

            if hasattr(self.c, "media_dir") and self.c.media_dir:
                command_parts.append(f'--media_dir="{self.c.media_dir}"')
            if hasattr(self.c, "image_max_pixels") and self.c.image_max_pixels:
                command_parts.append(f'--image_max_pixels="{self.c.image_max_pixels}"')

            child_env = os.environ.copy()
            child_env["CUDA_VISIBLE_DEVICES"] = "0"
            child_env["LLAMAFACTORY_VERBOSITY"] = "ERROR"

            process = subprocess.Popen(
                command_parts,
                env=child_env,
                stdout=None,  # Use None to indicate using parent process's stdout (i.e., terminal)
                stderr=None,
                text=True,
                bufsize=1,
            )  # nosec
            return_code = process.wait()
            if return_code != 0:
                logger.error(
                    f"Command '{' '.join(command_parts)}' execution failed with return code {return_code}"
                )
        except FileNotFoundError:
            logger.error(
                f"Command execution failed: executable '{command_parts[0]}' or script '{command_parts[1]}' not found"
            )
        except KeyError as e:
            logger.error(f"Failed to execute length_cdf.py script: missing configuration item {str(e)}")
        except Exception as e:
            logger.error(f"Unknown error occurred while executing length_cdf.py script: {str(e)}")

    def get_csv_files(self):
        """Traverse the folder to get all CSV file paths and sort by starting sequence number in filename"""

        csv_files = []
        for chat_obj_folder in os.listdir(self.csv_folder):
            chat_obj_folder_path = os.path.join(self.csv_folder, chat_obj_folder)
            for csvfile in os.listdir(chat_obj_folder_path):
                if not csvfile.endswith(".csv"):
                    continue
                csvfile_path = os.path.join(chat_obj_folder_path, csvfile)
                csv_files.append(csvfile_path)
        pattern = re.compile(r"_(\d+)_\d+\.csv$")

        def extract_start(fp: str) -> int:
            name = os.path.basename(fp)
            m = pattern.search(name)
            return int(m.group(1)) if m else 0

        csv_files.sort(key=extract_start)
        return csv_files

    def match_qa(self, messages: List[ChatMessage]) -> List[Union[QaPair, CutMessage]]:
        """
        Match question-answer pairs

        Args:
            messages: Message list

        Returns:
            List[Union[QaPair, CutMessage]]: List of Q&A pairs containing instructions and outputs
        """
        WAITING_INSTRUCTION = "waiting_instruction"
        WAITING_RESPONSE = "waiting_response"

        current_state = WAITING_INSTRUCTION
        qa_res: List[Union[QaPair, CutMessage]] = []
        last_message = None
        # Accumulate consecutive instructions from the other party instead of overwriting them,
        # so multi-message instructions (e.g. "在吗" ... "收到没") are all kept
        current_instructions: List[ChatMessage] = []
        qa_id_counter = 0

        conversation_messages: List[Message] = []
        conversation_images: List[str] = []
        conversation_talker = ""
        conversation_is_group = False

        def _calculate_qa_length(
            messages: List[Message], new_user_content: str, new_assistant_content: str
        ) -> int:
            """Calculate total character length of messages plus new messages"""
            total_length = 0
            for msg in messages:
                total_length += len(msg.content)
            total_length += len(new_user_content) + len(new_assistant_content)
            return total_length

        def _save_current_qa_pair(
            qa_id: int,
            time_stamp: Timestamp,
            current_conversation_messages: List[Message],
            current_conversation_images: List[str],
            talker: str = "",
            is_group: bool = False,
        ) -> int:
            """Helper function to save the current QA pair."""
            nonlocal qa_res  # Allow modification of qa_res from the outer scope

            total_length = _calculate_qa_length(current_conversation_messages, "", "")

            if total_length <= self.config.messages_max_length:
                if len(current_conversation_images) > self.config.max_image_num:
                    logger.warning(
                        f"QA pair (potential id {qa_id}) with timestamp {time_stamp} "
                        f"has too many images ({len(current_conversation_images)} > {self.config.max_image_num}) "
                        "and will be skipped."
                    )
                    return qa_id

                if (
                    len(current_conversation_messages) == 2
                    and current_conversation_messages[0].role == "user"
                    and current_conversation_messages[0].content == "<begin_chat>"
                ):
                    return qa_id

                system_content = self.system_prompt
                if self.c.add_time:
                    system_content += f"\n 现在时间是{time_stamp.strftime('%m-%d %H:%M')}"
                if self.c.add_relation and talker:
                    relation = self.relations.get(talker, "")
                    if relation:
                        system_content += f"\n 对方是你的{relation}，你们正在聊天"

                processed_messages = current_conversation_messages.copy()
                # Drop synthetic <begin_chat> instruction pairs. Injecting "你应该说：X"
                # into the user turn teaches the model a tag-extraction shortcut instead
                # of context-conditioned replies (a path dependency that never transfers
                # to real inference); the subsequent natural turns are kept.
                filtered_messages: List[Message] = []
                i = 0
                while i < len(processed_messages):
                    if (
                        processed_messages[i].role == "user"
                        and "<begin_chat>" in processed_messages[i].content
                        and i + 1 < len(processed_messages)
                        and processed_messages[i + 1].role == "assistant"
                    ):
                        i += 2  # drop the synthetic pair
                        continue
                    filtered_messages.append(processed_messages[i])
                    i += 1
                # A leading assistant turn has no context to condition on
                while filtered_messages and filtered_messages[0].role == "assistant":
                    filtered_messages.pop(0)
                if not filtered_messages:
                    return qa_id
                processed_messages = filtered_messages

                qa_pair = self.QaPair(
                    id=qa_id,
                    time=time_stamp,
                    score=0,
                    messages=processed_messages,
                    images=current_conversation_images.copy(),
                    system=system_content,
                    group=is_group,
                )
                qa_res.append(qa_pair)
                return qa_id + 1
            else:
                logger.warning(
                    f"QA pair (potential id {qa_id}) with timestamp {time_stamp} "
                    f"exceeds max length ({total_length} > {self.config.messages_max_length}) "
                    "and will be skipped."
                )
                return qa_id

        for msg in messages:
            if isinstance(msg, CutMessage):
                # When encountering CutMessage, save current conversation and reset state
                if conversation_messages:
                    qa_id_counter = _save_current_qa_pair(
                        qa_id_counter,
                        last_message.CreateTime if last_message else msg.CreateTime,
                        conversation_messages,
                        conversation_images,
                        conversation_talker,
                        conversation_is_group,
                    )
                # Reset state
                current_state = WAITING_INSTRUCTION
                current_instructions = []
                last_message = None
                conversation_messages = []
                conversation_images = []
                conversation_talker = ""
                conversation_is_group = False
                continue

            if current_state == WAITING_INSTRUCTION:
                if msg.is_sender == 0:  # Received message from other party
                    if last_message and not self.qa_match_strategy.is_same_conversation([last_message], msg):
                        # If not the same conversation and there is a previous message, save the previous conversation
                        if conversation_messages:
                            qa_id_counter = _save_current_qa_pair(
                                qa_id_counter,
                                last_message.CreateTime,
                                conversation_messages,
                                conversation_images,
                                conversation_talker,
                                conversation_is_group,
                            )
                            conversation_messages = []
                            conversation_images = []
                            conversation_is_group = False

                    # Regardless of whether a new conversation has just been started, this 'msg' now becomes the current instruction.
                    current_instructions = [msg]
                    last_message = msg
                    conversation_talker = msg.talker
                    current_state = WAITING_RESPONSE
                elif msg.is_sender == 1:  # Own message as first message
                    if last_message and not self.qa_match_strategy.is_same_conversation([last_message], msg):
                        if conversation_messages:
                            qa_id_counter = _save_current_qa_pair(
                                qa_id_counter,
                                last_message.CreateTime,
                                conversation_messages,
                                conversation_images,
                                conversation_talker,
                                conversation_is_group,
                            )
                            conversation_messages = []
                            conversation_images = []
                            conversation_is_group = False

                    conversation_messages.append(Message(role="user", content="<begin_chat>"))
                    conversation_messages.append(Message(role="assistant", content=msg.msg))
                    conversation_is_group = conversation_is_group or msg.is_group
                    last_message = msg

            elif current_state == WAITING_RESPONSE:
                if msg.is_sender == 0:  # Received message from other party
                    if last_message and not self.qa_match_strategy.is_same_conversation([last_message], msg):
                        if conversation_messages:
                            qa_id_counter = _save_current_qa_pair(
                                qa_id_counter,
                                last_message.CreateTime,
                                conversation_messages,
                                conversation_images,
                                conversation_talker,
                                conversation_is_group,
                            )
                            conversation_messages = []
                            conversation_images = []
                            conversation_is_group = False
                    # Accumulate instead of overwriting: consecutive messages from the other
                    # party all become part of the pending instruction (previously the earlier
                    # instruction was silently discarded here)
                    current_instructions.append(msg)
                    last_message = msg
                    conversation_talker = msg.talker
                    # State remains unchanged
                else:  # Own message - use strategy to determine if it belongs to the same conversation
                    if last_message and self.qa_match_strategy.is_same_conversation([last_message], msg):
                        if not current_instructions:
                            raise ValueError("current_instructions should not be empty when creating a QA pair")

                        # All accumulated instructions become one user turn
                        instruction_content = "\n".join(instruction.msg for instruction in current_instructions)
                        conversation_messages.append(Message(role="user", content=instruction_content))
                        conversation_messages.append(Message(role="assistant", content=msg.msg))
                        conversation_is_group = conversation_is_group or any(
                            instruction.is_group for instruction in current_instructions
                        )
                        for current_instruction in current_instructions:
                            if hasattr(current_instruction, "src") and current_instruction.src:
                                if isinstance(current_instruction.src, list):
                                    valid_images = [img_src for img_src in current_instruction.src if img_src]
                                    if valid_images:
                                        conversation_images.extend(valid_images)
                                elif current_instruction.src:
                                    conversation_images.append(current_instruction.src)
                        last_message = msg
                    else:
                        # Own reply outside the time window: previously it was silently dropped and
                        # the conversation was neither saved nor cleared. Save the conversation first,
                        # then keep the reply as a self-initiated turn instead of losing it.
                        if conversation_messages:
                            qa_id_counter = _save_current_qa_pair(
                                qa_id_counter,
                                last_message.CreateTime if last_message else msg.CreateTime,
                                conversation_messages,
                                conversation_images,
                                conversation_talker,
                                conversation_is_group,
                            )
                            conversation_messages = []
                            conversation_images = []
                            conversation_is_group = False
                        conversation_messages.append(Message(role="user", content="<begin_chat>"))
                        conversation_messages.append(Message(role="assistant", content=msg.msg))
                        conversation_is_group = conversation_is_group or msg.is_group
                        last_message = msg

                    # Regardless of whether it matches, reset state
                    current_state = WAITING_INSTRUCTION
                    current_instructions = []

        # Process the last conversation
        if conversation_messages and last_message:
            qa_id_counter = _save_current_qa_pair(
                qa_id_counter,
                last_message.CreateTime,
                conversation_messages,
                conversation_images,
                conversation_talker,
                conversation_is_group,
            )

        return qa_res

    def group_consecutive_messages(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """
        Combine multiple consecutive messages from the same person into one message, add cut when encountering cut_type

        Args:
            messages: Message list

        Returns:
            List[ChatMessage]: Combined message list
        """
        if not messages:
            return []

        def _combine_text(messages: List[ChatMessage]) -> ChatMessage:
            """
            Merge multiple messages into one

            Args:
                messages: List of messages to merge

            Returns:
                ChatMessage: Merged message
            """
            base_msg = messages[0]
            combined_content = messages[0].msg
            combined_src_list = [messages[0].src] if messages[0].modality == DataModality.IMAGE else []

            for i in messages[1:]:
                content = i.msg
                if not content:
                    continue

                if combined_content and combined_content[-1] not in [
                    "。",
                    ".",
                    "！",
                    "!",
                    "？",
                    "?",
                    "…",
                    "，",
                    ",",
                ]:
                    combined_content += "\n"

                if i.modality == DataModality.IMAGE:
                    combined_src_list.append(i.src)

                combined_content += content

            if len(combined_content) > self.c.combine_msg_max_length:
                logger.warning(
                    f"Combined message length exceeds {self.c.combine_msg_max_length}, will truncate: {combined_content[:50]}"
                )
                combined_content = combined_content[: self.c.combine_msg_max_length]
                remaining_image_count = combined_content.count("<image>")
                if len(combined_src_list) > remaining_image_count:
                    combined_src_list = combined_src_list[:remaining_image_count]

            combined_message = ChatMessage(
                id=base_msg.id,
                MsgSvrID=base_msg.MsgSvrID,
                type_name=base_msg.type_name,
                is_sender=base_msg.is_sender,
                talker=base_msg.talker,
                room_name=base_msg.room_name,
                msg=combined_content,
                src=combined_src_list,  # type: ignore
                CreateTime=messages[-1].CreateTime,  # Use the time of the last message
                modality=base_msg.modality,
                is_forward=base_msg.is_forward,
                is_group=base_msg.is_group,
            )

            return combined_message

        def _create_cut_message(message: ChatMessage) -> CutMessage:
            return CutMessage(
                is_sender=message.is_sender,
                cut_type=message.type_name,
                CreateTime=message.CreateTime,
            )

        def _combine_current_group(group):
            """
            Process current message group and add to grouped_messages

            Args:
                group: Current message group
            """
            if len(group) > 1:
                combined_msg = _combine_text(group)
                grouped_messages.append(combined_msg)
            else:
                grouped_messages.append(group[0])

        grouped_messages = []
        current_group = []

        for _, current_msg in enumerate(messages):
            if current_msg.type_name in self.cut_type_list or (
                current_msg.modality == DataModality.IMAGE and current_msg.is_sender == 1
            ):  # Own image messages need to be cut
                if current_group:
                    # Current group has messages, combine current group and add a cut
                    _combine_current_group(current_group)
                    current_group = []

                    cut_msg = _create_cut_message(current_msg)
                    grouped_messages.append(cut_msg)
                else:
                    # Current group has no messages, check previous group
                    if grouped_messages:
                        if not isinstance(grouped_messages[-1], CutMessage):
                            cut_msg = _create_cut_message(current_msg)
                            grouped_messages.append(cut_msg)
                    # If previous group has no messages or last one is CutMessage, continue directly
                continue

            if not current_group:
                current_group = [current_msg]
                continue

            last_msg = current_group[-1]

            # Determine if it's consecutive messages from the same person
            if (
                current_msg.is_sender == last_msg.is_sender
                and current_msg.talker == last_msg.talker
                and self.single_combine_strategy.is_same_conversation([last_msg], current_msg)
            ):
                current_group.append(current_msg)
            else:
                # Not messages from the same person, process current group and start new group
                _combine_current_group(current_group)
                # Start new group
                current_group = [current_msg]

        # Process the last group of messages
        if current_group:
            _combine_current_group(current_group)

        return grouped_messages

    def process_by_msgtype(self, chat_message: ChatMessage):
        if chat_message.type_name.lower() in ["文本", "text"]:
            self.process_text(chat_message)
        # elif chat_message.modality == DataModality.IMAGE:
        #     self.process_image(chat_message)

    # chatlog exports inline non-text message types as bracketed placeholders in text rows
    _pure_placeholder_re = re.compile(
        r"^(\[(语音通话|视频通话|动画表情|QQ红包|位置|文件|位置共享|小程序|音乐|图片|链接|视频|聊天记录|卡片链接|表情|戳一戳)\])$"
    )
    # call records carry a duration ("[语音通话] 32:03") or a status ("[语音通话] 已取消")
    _pure_call_re = re.compile(r"^\[(语音通话|视频通话)\](?:\s+\d{1,3}:\d{2}(?::\d{2})?|\s+已取消)?$")
    _inline_call_re = re.compile(r"\[(语音通话|视频通话)\](?:\s+\d{1,3}:\d{2}(?::\d{2})?|\s+已取消)?")
    _inline_placeholder_re = re.compile(r"\[(图片|视频|表情|动画表情|链接|文件|音乐|小程序|QQ红包|卡片链接|聊天记录|转发的聊天记录|语音|名片|消息|闪照|表情包|通话|Photo|Audio)\]")
    _quote_re = re.compile(r"\[引用\s*[^：:\]]{0,30}[：:]([^\]]*)\]")
    _control_char_re = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
    # QQ emoji text codes: /表情名 (new exports) and [表情名] (old exports) are deleted
    _emoji_names = sorted(QQ_EMOJI_CODES, key=len, reverse=True)
    _emoji_alt = "|".join(re.escape(name) for name in _emoji_names)
    _slash_emoji_re = re.compile(r"/(%s)" % _emoji_alt)
    _bracket_emoji_re = re.compile(r"\[(%s)\]" % _emoji_alt)

    def clean_chat_text(self, raw: str):
        """
        Clean a single chat message:
        - strips control characters (e.g. \\x14 left by group exports)
        - drops pure-placeholder / system messages ([语音通话], 撤回了一条消息, 你已添加了...)
        - keeps quoted-reply content but removes the quoted person's nickname (privacy)
        - anonymizes @mentions via the global nickname map built by _build_nickname_map
        - converts inline placeholders such as [图片] to natural Chinese (（图片）)

        Returns the cleaned text, or None if the message should be dropped entirely.
        """
        text = self._control_char_re.sub("", raw).strip()
        if not text:
            return None
        # Pure placeholders and system-generated notices carry no conversational value
        if self._pure_placeholder_re.match(text) or self._pure_call_re.match(text):
            return None
        if "撤回了一条消息" in text:
            return None
        if text.startswith("你已添加了") and len(text) <= 40:
            return None
        if "以上是打招呼的消息" in text or "我通过了你的朋友验证请求" in text:
            return None
        if text == "微信红包":
            return None
        if text.startswith("[自动回复]"):
            return None
        if text.startswith(("[转账]", "[转账收款]")):
            return None
        if text.startswith("<msg>") or "<appmsg" in text:
            # raw XML blob leaked from file/attachment messages
            return None
        if "你有一笔待接收的转账" in text:
            # payment platform notice embedded in chat text
            text = text.replace("你有一笔待接收的转账。", "").replace("你有一笔待接收的转账", "")
        # Delete QQ emoji text codes (/呲牙, [捂脸] ...) so the model never learns
        # to emit unrenderable slash/bracket codes instead of real emoji
        text = self._slash_emoji_re.sub("", text)
        text = self._bracket_emoji_re.sub("", text)
        # Strip inline call markers ("嗯？[语音通话] 00:14" -> "嗯？")
        text = self._inline_call_re.sub("", text)
        # [引用 昵称：内容] -> [引用] 内容 (drop the nickname for privacy, keep replied-to content)
        text = self._quote_re.sub(r"[引用] \1", text)
        # @nickname -> @我 / @联系人N
        if self._mention_re is not None:
            text = self._mention_re.sub(lambda m: m.group(1) + self.name_map.get(m.group(2), m.group(2)), text)
        # Inline placeholders -> natural Chinese
        text = self._inline_placeholder_re.sub(r"（\1）", text)
        return text.strip() or None

    def _build_nickname_map(self, csv_files: List[str]) -> None:
        """
        Pre-scan all CSVs to collect nicknames that appear in @mentions, so that the
        same real name is always mapped to the same pseudonym across the whole dataset.
        Names are only collected when mentioned at least twice (avoids matching things
        like @media in code snippets); the user's own names are mapped to "我".
        """
        from collections import Counter

        mention_counter: "Counter[str]" = Counter()
        self_names_counter: "Counter[str]" = Counter()
        candidate_pattern = re.compile(r"[@＠]\s*([A-Za-z0-9_一-鿿぀-ヿ가-힯·\-]{1,24})")

        for fp in csv_files:
            try:
                df = pd.read_csv(fp, encoding="utf-8", dtype={"msg": str, "src": str}, keep_default_na=False)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to pre-scan CSV {fp}: {e}")
                continue
            text_mask = df["type_name"].astype(str).str.lower().isin(["文本", "text"])
            for msg in df.loc[text_mask, "msg"]:
                for m in candidate_pattern.finditer(str(msg)):
                    mention_counter[m.group(1)] += 1
            own_mask = df["is_sender"] == 1
            for talker in df.loc[own_mask, "talker"]:
                self_names_counter[str(talker)] += 1

        self_names = {name for name, count in self_names_counter.most_common() if name and count >= 2}
        mentioned_names = [
            name for name, count in mention_counter.most_common() if count >= 2 and name != "全体成员"
        ]

        self.name_map: dict = {}
        contact_index = 0
        for name in mentioned_names:
            if name in self_names:
                self.name_map[name] = "我"
            else:
                contact_index += 1
                self.name_map[name] = f"联系人{contact_index}"

        if self.name_map:
            alternation = "|".join(re.escape(name) for name in sorted(self.name_map, key=len, reverse=True))
            self._mention_re = re.compile(r"([@＠])(%s)" % alternation)
        else:
            self._mention_re = None
        logger.info(f"Built @mention anonymization map: {len(self.name_map)} nicknames -> pseudonyms")

    def load_file(self, file_path) -> List[ChatMessage]:
        """
        Perform overall first preprocessing, filter rows that don't meet conditions, check if images exist and change type to cut if not, add DataModality field
        """
        folder_path = os.path.dirname(file_path)
        folder_name = os.path.basename(folder_path)
        is_group = folder_name.startswith("QQ群")  # chatlog export: group chat folders use the QQ群 prefix

        if folder_name not in self.relations:
            users_json_path = os.path.join(folder_path, "users.json")
            if os.path.exists(users_json_path):
                try:
                    with open(users_json_path, encoding="utf-8") as f:
                        users_data = json.load(f)
                        relation = users_data.get("relation", "")
                        if relation:
                            self.relations[folder_name] = relation
                            logger.debug(f"Loaded relation for {folder_name}: {relation}")
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    logger.warning(f"Failed to load users.json from {folder_path}: {e}")

        df = pd.read_csv(
            file_path,
            encoding="utf-8",
            dtype={"msg": str, "src": str},
            escapechar=None,
            keep_default_na=False,
        )

        df = df[~df["type_name"].isin(values=self.skip_type_list)]

        if "is_forward" in df.columns:
            df = df[~((df["is_sender"] == 1) & (df["is_forward"]))]

        # Text-level cleaning: strip control chars, drop pure-placeholder/system messages,
        # anonymize @mentions and quoted-reply nicknames
        clean_drop_indices = []
        for i in df.index:
            if df.loc[i, "type_name"].lower() in ["文本", "text"]:  # type: ignore
                cleaned = self.clean_chat_text(str(df.loc[i, "msg"]))
                if cleaned is None:
                    clean_drop_indices.append(i)
                else:
                    df.loc[i, "msg"] = cleaned
        if clean_drop_indices:
            df = df.drop(index=clean_drop_indices)

        # Batch anonymize PII in text messages (mask sensitive spans instead of dropping
        # whole messages, so paired Q&A data is not destroyed) and check blocked words
        text_indices = []
        text_messages = []

        for i in df.index:
            if df.loc[i, "type_name"].lower() in ["文本", "text"]:  # type: ignore
                text_indices.append(i)
                text_messages.append(str(df.loc[i, "msg"]))

        indices_to_drop = []
        if text_messages:
            anonymized_texts = self.pii_detector.anonymize_batch(text_messages)

            for df_index, msg_str, anonymized in zip(text_indices, text_messages, anonymized_texts):
                df.loc[df_index, "msg"] = anonymized

                # Check blocked words
                for blocked_word in self.blocked_words:
                    if blocked_word in msg_str:
                        indices_to_drop.append(df_index)
                        break

        if indices_to_drop:
            df = df.drop(index=indices_to_drop)

        # Process other message types
        for i in df.index:
            if df.loc[i, "type_name"].lower() in ["文本", "text"]:
                continue
            if df.loc[i, "src"].lower().endswith(".gif"):
                df.loc[i, "src"] = ""
                df.loc[i, "type_name"] = "动画表情" if self.c.platform == PlatformType.CHAT else "sticker"
                continue
            if df.loc[i, "type_name"].lower() in ["图片", "image"]:  # type: ignore
                if self.c.platform in [PlatformType.CHAT, PlatformType.TELEGRAM]:
                    result = check_image_file_exists(str(df.loc[i, "src"]))
                    if isinstance(result, str) and df.loc[i, "is_sender"] == 0:
                        df.loc[i, "src"] = result
                        df.loc[i, "msg"] = "<image>"
                        df.loc[i, "modality"] = DataModality.IMAGE
                    else:
                        df.loc[i, "type_name"] = "Cut"
            elif df.loc[i, "type_name"] in ["sticker", "动画表情"]:
                if self.c.platform in [PlatformType.CHAT, PlatformType.TELEGRAM]:
                    df.loc[i, "src"] = ""
                    continue
            else:
                df.loc[i, "msg"] = ""

        df = df.dropna(how="all")
        # Time format: 2021-07-07 10:27:23
        df["CreateTime"] = pd.to_datetime(df["CreateTime"])

        return [ChatMessage(**row, is_group=is_group) for row in df.to_dict("records")]  # type: ignore

    def process_text(self, chat_message: ChatMessage):
        pass

    def save_result(self, qa_res: List[QaPair]):
        """
        Saves the list of QaPair objects to a JSON file after converting them to dictionaries.

        Args:
            qa_res: A list of QaPair objects.
        """
        processed_qa_res = []
        for idx, item in enumerate(qa_res):
            item_dict = {
                "id": str(idx),
                "time": item.time.isoformat() if item.time else None,
                "score": item.score,
                "messages": [{"role": msg.role, "content": msg.content} for msg in item.messages],
                "images": item.images,
                "system": item.system,
                "group": item.group,
            }
            processed_qa_res.append(item_dict)

        output_path = "./dataset/res_csv/sft/sft-my.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed_qa_res, f, ensure_ascii=False, indent=4)
        logger.success(
            f"Chat record processing successful, {len(qa_res)} entries in total, saved to {output_path}"
        )


if __name__ == "__main__":
    processor = DataProcessor()
    processor.main()
