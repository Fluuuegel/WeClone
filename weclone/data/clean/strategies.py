import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, cast

import pandas as pd
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm

from weclone.core.inference.online_infer import OnlineLLM
from weclone.data.models import QaPair, QaPairScore, QaPairScoreWithId
from weclone.data.qq_emojis import QQ_EMOJI_CODES
from weclone.prompts.clean_data import CLEAN_PROMPT
from weclone.utils.config_models import WCMakeDatasetConfig
from weclone.utils.log import logger


class ChatTextCleaner:
    """Regex-based text cleaning for chatlog-exported chat messages."""

    # 括号字符集：兼容中英文方括号、圆括号（_R_BR_CHARS 不带类括号，用于嵌入其他字符类）
    _L_BR = r"[\[［（(]"
    _R_BR = r"[\]］）)]"
    _R_BR_CHARS = r"\]］）)"

    # 纯占位符整条消息 → 丢弃（含通话时长/状态形态）
    _pure_placeholder_re = re.compile(
        rf"^{_L_BR}(语音通话|视频通话|动画表情|QQ红包|位置|文件|位置共享|小程序|音乐|图片|链接|视频|聊天记录|卡片链接|表情|戳一戳|闪照){_R_BR}$"
    )
    _pure_call_re = re.compile(rf"^{_L_BR}(语音通话|视频通话){_R_BR}(?:\s+\d{{1,3}}:\d{{2}}(?::\d{{2}})?|\s+已取消)?$")
    _inline_call_re = re.compile(rf"{_L_BR}(语音通话|视频通话){_R_BR}(?:\s+\d{{1,3}}:\d{{2}}(?::\d{{2}})?|\s+已取消)?")
    # 行内占位符 → 全角括号（[图片] → （图片））
    _inline_placeholder_re = re.compile(
        rf"{_L_BR}(图片|视频|表情|动画表情|链接|文件|音乐|小程序|卡片链接|聊天记录|转发的聊天记录|语音|消息|表情包|Photo|Audio){_R_BR}"
    )
    # 无转换价值的占位符：任何位置都直接删除
    _always_delete_placeholder_re = re.compile(rf"{_L_BR}(闪照|名片|通话|QQ红包){_R_BR}")
    # 系统长句："（链接） 邀请你加入群聊" / "XX参与了接龙"
    _system_invite_re = re.compile(rf"{_L_BR}(链接|小程序|群聊){_R_BR}\s*(邀请你加入群聊|.*参与了接龙).*")
    # 消息开头的分享占位符+标题 → 整条丢弃；句中的 [链接] 不受影响
    _share_card_re = re.compile(rf"^{_L_BR}(链接|小程序|卡片链接){_R_BR}\s*\S")
    # 付款诈骗话术：以固定短语结尾 → 整条丢弃
    _payment_scam_re = re.compile(r"就可以帮我付款啦[\s，。！？!?.]*$")
    # QQ 客户端闪照提示语：逐句删除，保留前后用户文字
    _flash_photo_notice_re = re.compile(rf"{_L_BR}闪照{_R_BR}\s*请使用新版手机QQ查看闪照[。.]?\s*")
    # 引用整体删除（末尾可选右括号覆盖嵌套 [图片]）；未闭合引用删到消息末尾
    _quote_re = re.compile(rf"{_L_BR}引用\s*[^：:]{{0,30}}[：:][^{_R_BR_CHARS}]*{_R_BR}{_R_BR}?")
    _quote_open_re = re.compile(rf"{_L_BR}引用\s*[^：:]{{0,30}}[：:][^{_R_BR_CHARS}]*$")
    _auto_reply_re = re.compile(rf"^{_L_BR}自动回复{_R_BR}")
    _transfer_re = re.compile(rf"^{_L_BR}(转账|转账收款){_R_BR}")
    _control_char_re = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
    _url_re = re.compile(r"https?://[^\s\\]*(?:\n)?")
    # QQ 表情代码（/呲牙、[捂脸]）：白名单命中即删除
    _emoji_names = sorted(QQ_EMOJI_CODES, key=len, reverse=True)
    _emoji_alt = "|".join(re.escape(name) for name in _emoji_names)
    _slash_emoji_re = re.compile(r"/(%s)" % _emoji_alt)
    _bracket_emoji_re = re.compile(rf"{_L_BR}(%s){_R_BR}" % _emoji_alt)

    def __init__(self):
        self.name_map: Dict[str, str] = {}
        self._mention_re: Optional[re.Pattern] = None

    def set_mention_map(self, name_map: Dict[str, str]) -> None:
        """Install the global @mention anonymization map (name -> pseudonym)."""
        self.name_map = name_map
        if name_map:
            alternation = "|".join(re.escape(name) for name in sorted(name_map, key=len, reverse=True))
            self._mention_re = re.compile(r"([@＠])(%s)" % alternation)
        else:
            self._mention_re = None

    def clean(self, raw: str) -> Optional[str]:
        """清洗单条消息：返回清洗后文本；应整条丢弃的消息返回 None。"""
        text = self._control_char_re.sub("", raw).strip()
        if not text:
            return None
        # 纯占位符/系统消息：整条丢弃
        if self._pure_placeholder_re.match(text) or self._pure_call_re.match(text):
            return None
        if "撤回了一条消息" in text:
            return None
        if "转发的聊天记录" in text:
            return None
        if text.startswith("你已添加了") and len(text) <= 40:
            return None
        if "以上是打招呼的消息" in text or "我通过了你的朋友验证请求" in text:
            return None
        if text == "微信红包":
            return None
        if self._auto_reply_re.match(text):
            return None
        if self._transfer_re.match(text):
            return None
        if text.startswith("<msg>") or "<appmsg" in text:
            return None
        if "你有一笔待接收的转账" in text:
            text = text.replace("你有一笔待接收的转账。", "").replace("你有一笔待接收的转账", "")
        # 分享卡片/付款话术：必须在 URL 删除与占位符转换之前判断
        if self._share_card_re.match(text):
            return None
        if self._payment_scam_re.search(text):
            return None
        text = self._url_re.sub("", text)
        text = self._slash_emoji_re.sub("", text)
        text = self._bracket_emoji_re.sub("", text)
        text = self._inline_call_re.sub("", text)
        text = self._quote_re.sub("", text)
        text = self._quote_open_re.sub("", text)
        # 系统长句/闪照提示语/无价值占位符：必须在行内占位符转换之前执行
        text = self._system_invite_re.sub("", text)
        text = self._flash_photo_notice_re.sub("", text)
        text = self._always_delete_placeholder_re.sub("", text)
        # @昵称 → @我 / @联系人N
        if self._mention_re is not None:
            text = self._mention_re.sub(lambda m: m.group(1) + self.name_map.get(m.group(2), m.group(2)), text)
        text = self._inline_placeholder_re.sub(r"（\1）", text)
        return text.strip() or None


@dataclass
class CleaningStrategy(ABC):
    """Abstract base class for data cleaning strategies, but provides common cleaning methods"""

    make_dataset_config: WCMakeDatasetConfig

    @abstractmethod
    def judge(self, data: List[QaPair]) -> None:
        """
        Scoring method, needs to be implemented by subclasses.
        """
        pass

    def clean(self) -> str:
        """
        Filter SFT data based on score and return the final dataset name to use.
        """
        config = self.make_dataset_config
        original_dataset_name = config.dataset
        cleaned_dataset_name = original_dataset_name + "-cleaned"

        dataset_dir = config.dataset_dir
        dataset_info_path = os.path.join(dataset_dir, "dataset_info.json")

        with open(dataset_info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        paths = {}
        for name in [original_dataset_name, cleaned_dataset_name]:
            file_name = info.get(name, {}).get("file_name")
            if not file_name:
                logger.error(
                    f"Dataset '{name}' is not defined in dataset_info.json, will use original dataset."
                )
                return original_dataset_name
            paths[name] = os.path.join(dataset_dir, file_name)
        original_data_path, cleaned_data_path = paths.values()

        try:
            with open(original_data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            accept_score = config.clean_dataset.llm.accept_score
            filtered_data = [item for item in data if item.get("score", 0) >= accept_score]

            if not filtered_data:
                logger.warning("No data retained after cleaning, will use original dataset.")
                return original_dataset_name

            with open(cleaned_data_path, "w", encoding="utf-8") as f:
                json.dump(filtered_data, f, ensure_ascii=False, indent=2)
            logger.success(
                f"Filtered data below {accept_score} score, retained {len(filtered_data)} items, saved to {cleaned_data_path}"
            )
            return cleaned_dataset_name

        except Exception as e:
            logger.error(f"Error occurred during data cleaning, will use original dataset: {e}")
            return original_dataset_name


@dataclass
class LLMCleaningStrategy(CleaningStrategy):
    """Strategy for data cleaning using large language models"""

    make_dataset_config: WCMakeDatasetConfig

    def judge(self, data: List[QaPair]) -> None:
        """
        Call LLM for scoring and directly assign scores to the input QaPair.
        """
        from weclone.core.inference.offline_infer import vllm_infer

        logger.info("Starting LLM scoring of group chat data")
        inputs = []
        prompt_template = PromptTemplate.from_template(CLEAN_PROMPT)
        for qa in data:
            if qa.images or not qa.group:
                # Only group-chat records are scored (they are prone to semantic incoherence
                # from multi-talker stitching); private chats and image records bypass cleaning
                qa.score = 6
            else:
                messages_str = ""
                for msg in qa.messages:
                    if msg.role == "user":
                        messages_str += f"Q: {msg.content}\n"
                    elif msg.role == "assistant":
                        messages_str += f"A: {msg.content}\n"
                prompt_value = prompt_template.invoke({"id": qa.id, "messages": messages_str.strip()})
                inputs.append(prompt_value.to_string())

        parsed_scores, failed_indexs = vllm_infer(
            inputs,
            self.make_dataset_config.model_name_or_path,
            template=self.make_dataset_config.template,
            temperature=0,
            guided_decoding_class=QaPairScore,
            repetition_penalty=1.1,
            enable_thinking=self.make_dataset_config.clean_dataset.llm.enable_thinking,
            cutoff_len=self.make_dataset_config.messages_max_length + 1024,  # add prompt length
            max_new_tokens=1024 if self.make_dataset_config.clean_dataset.llm.enable_thinking else 200,
        )

        # We align scores by iterating only scored (group-chat) examples and popping from the head of parsed_scores.
        # Build an iterator over parsed results for simplicity and safety.
        parsed_iter = iter(cast(List[QaPairScore | None], parsed_scores))
        scored_count = 0
        failed_count = 0

        for qa in data:
            if qa.images or not qa.group:
                continue
            scored_count += 1
            parsed_item = next(parsed_iter, None)
            if parsed_item is None:
                failed_count += 1
                qa.score = 0
            else:
                qa.score = parsed_item.score

        # Sanity check: number of Nones should equal failed_indexs; and total length matches scored count
        assert failed_count == len(failed_indexs), (
            f"Mismatch: failed_count({failed_count}) != failed_indexs({len(failed_indexs)})"
        )
        assert len(cast(List[QaPairScore | None], parsed_scores)) == scored_count, (
            f"Mismatch: len(parsed_scores)({len(cast(List[QaPairScore | None], parsed_scores))}) != scored_count({scored_count})"
        )

        scores = [qa.score for qa in data if qa.score is not None]
        score_series = pd.Series(scores)
        score_counts = score_series.value_counts().sort_index()
        score_percentages = score_series.value_counts(normalize=True).sort_index() * 100
        pd.set_option("display.unicode.east_asian_width", True)  # Try to fix alignment issues
        distribution_df = pd.DataFrame(  # Merge count and percentage into one DataFrame for printing
            {
                "Count": score_counts,
                "Percentage(%)": score_percentages.round(2),
            }
        )
        distribution_df.index.name = "Score"  # Add column name for the first column: Score
        printable_df_str = distribution_df.reset_index().to_string(index=False)
        logger.success(f"LLM scoring distribution:\n{printable_df_str}")


@dataclass
class OlineLLMCleaningStrategy(CleaningStrategy):
    """Strategy for data cleaning using large language models"""

    # TODO: images clean support
    def judge(self, data: List[QaPair]) -> None:
        config = self.make_dataset_config
        logger.info("Starting online model scoring of group chat data")
        logger.info(f"Using model {config.model_name}")

        client = OnlineLLM(
            api_key=config.llm_api_key,
            base_url=config.base_url,
            model_name=config.model_name,
            max_workers=config.clean_batch_size + 5,
        )

        inputs = []
        prompt_template = PromptTemplate.from_template(CLEAN_PROMPT)
        for qa in data:
            if qa.images or not qa.group:
                # Only group-chat records are scored; private chats and image records bypass cleaning
                qa.score = 6
            else:
                messages_str = ""
                for msg in qa.messages:
                    if msg.role == "user":
                        messages_str += f"Q: {msg.content}\n"
                    elif msg.role == "assistant":
                        messages_str += f"A: {msg.content}\n"
                prompt_value = prompt_template.invoke({"id": qa.id, "messages": messages_str.strip()})
                inputs.append(prompt_value.to_string())

        clean_batch_size = config.clean_batch_size
        all_parsed_scores = []

        for i in tqdm(range(0, len(inputs), clean_batch_size), desc="Online model scoring progress"):
            batch = inputs[i : i + clean_batch_size]

            try:
                parsed_results, failed_indexs = client.chat_batch(
                    batch, temperature=0, guided_decoding_class=QaPairScoreWithId
                )

                for j, parsed_result in enumerate(parsed_results):
                    if parsed_result is not None:
                        all_parsed_scores.append(parsed_result)
                    else:
                        logger.warning(f"Failed to parse result for batch item at index {i + j}")

            except Exception as e:
                logger.error(
                    f"Failed to call online model or parse result for batch starting at index {i}, error: {str(e)}"
                )

        score_map = {score.id: score.score for score in all_parsed_scores}
        for qa in data:
            if qa.images or not qa.group:
                continue
            if qa.id in score_map:
                qa.score = score_map[qa.id]
            else:
                logger.warning(f"No score obtained for QA ID {qa.id}, default assigned 0")
                qa.score = 0

        scores = [qa.score for qa in data if qa.score is not None]
        score_series = pd.Series(scores)
        score_counts = score_series.value_counts().sort_index()
        score_percentages = score_series.value_counts(normalize=True).sort_index() * 100
        pd.set_option("display.unicode.east_asian_width", True)
        distribution_df = pd.DataFrame(
            {
                "Count": score_counts,
                "Percentage(%)": score_percentages.round(2),
            }
        )
        distribution_df.index.name = "Score"
        printable_df_str = distribution_df.reset_index().to_string(index=False)
        logger.success(f"Online model scoring distribution:\n{printable_df_str}")
