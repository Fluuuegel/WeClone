"""Unit tests for ChatTextCleaner (weclone/data/clean/strategies.py)."""

import pytest

from weclone.data.clean.strategies import ChatTextCleaner


@pytest.fixture
def cleaner() -> ChatTextCleaner:
    return ChatTextCleaner()


class TestShareCardDrop:
    """Bug 1: message-start share placeholder + title text should be dropped entirely."""

    def test_link_placeholder_with_title_is_dropped(self, cleaner):
        assert cleaner.clean('[链接] 我校首届"璀璨时光"校园大学生超算竞赛拉开序幕！') is None

    def test_miniprogram_placeholder_with_title_is_dropped(self, cleaner):
        assert cleaner.clean("[小程序] 附近的人都在玩这款游戏") is None

    def test_card_link_placeholder_with_title_is_dropped(self, cleaner):
        assert cleaner.clean("（卡片链接） 某讲座报名入口") is None

    def test_fullwidth_link_placeholder_with_title_is_dropped(self, cleaner):
        assert cleaner.clean("（链接） 校园歌手大赛报名开始") is None

    def test_placeholder_only_message_still_dropped(self, cleaner):
        assert cleaner.clean("[链接]") is None

    def test_mid_sentence_placeholder_is_converted_not_dropped(self, cleaner):
        # regression: [链接] mid-sentence keeps the （链接） conversion semantics
        assert cleaner.clean("我发了个[链接]给你看看") == "我发了个（链接）给你看看"


class TestPaymentScamDrop:
    """Bug 2: messages ending with the fixed scam closing phrase should be dropped."""

    def test_scam_message_is_dropped(self, cleaner):
        assert cleaner.clean("411 http:，覆置 zr:/CIptAOd76eR咑亓🌟支..fu宝🌟就可以帮我付款啦") is None

    def test_scam_message_with_trailing_punctuation_is_dropped(self, cleaner):
        assert cleaner.clean("某某乱码就可以帮我付款啦！！") is None

    def test_phrase_inside_message_but_not_at_end_is_kept(self, cleaner):
        # regression: only the trailing occurrence identifies the scam notice
        text = '他说"就可以帮我付款啦"这句话很可疑'
        assert cleaner.clean(text) == text


class TestFlashPhotoNotice:
    """闪照 placeholder and its QQ client notice must be dropped."""

    def test_pure_flash_photo_placeholder_is_dropped(self, cleaner):
        assert cleaner.clean("（闪照）") is None

    def test_halfwidth_flash_photo_placeholder_is_dropped(self, cleaner):
        assert cleaner.clean("[闪照]") is None

    def test_flash_photo_notice_whole_message_is_dropped(self, cleaner):
        assert cleaner.clean("（闪照）请使用新版手机QQ查看闪照。") is None

    def test_halfwidth_flash_photo_notice_is_dropped(self, cleaner):
        assert cleaner.clean("[闪照]请使用新版手机QQ查看闪照。") is None

    def test_repeated_notices_keep_user_text(self, cleaner):
        # three notices + real user text: notices deleted, user text kept
        text = "（闪照）请使用新版手机QQ查看闪照。（闪照）请使用新版手机QQ查看闪照。（闪照）请使用新版手机QQ查看闪照。混合类型吧"
        assert cleaner.clean(text) == "混合类型吧"

    def test_flash_photo_with_user_comment_deletes_placeholder(self, cleaner):
        # bare （闪照） is ALWAYS deleted; the user's own comment stays
        assert cleaner.clean("（闪照）\n热疯了") == "热疯了"

    def test_flash_photo_mid_message_deletes_placeholder(self, cleaner):
        assert cleaner.clean("给你看一下\n（闪照）") == "给你看一下"

    def test_flash_photo_between_user_text_deletes_placeholder(self, cleaner):
        # placeholder removed; the surrounding newlines remain as-is
        assert cleaner.clean("片可以拆卸的\n（闪照）\n还凑活") == "片可以拆卸的\n\n还凑活"


class TestAlwaysDeletePlaceholders:
    """（名片）/（通话）/（QQ红包） follow the same always-delete logic as （闪照）."""

    def test_pure_card_placeholder_is_dropped(self, cleaner):
        assert cleaner.clean("[名片]") is None

    def test_pure_call_placeholder_is_dropped(self, cleaner):
        assert cleaner.clean("（通话）") is None

    def test_card_placeholder_with_user_text_deletes_placeholder(self, cleaner):
        assert cleaner.clean("[名片]这是我的名片，请惠存") == "这是我的名片，请惠存"

    def test_call_placeholder_with_user_text_deletes_placeholder(self, cleaner):
        assert cleaner.clean("给你看下[通话]记录") == "给你看下记录"

    def test_red_packet_with_greeting_deletes_placeholder(self, cleaner):
        assert cleaner.clean("[QQ红包]元旦快乐") == "元旦快乐"

    def test_image_and_link_conversion_unaffected(self, cleaner):
        # regression: convertible placeholders keep their （…） conversion behavior
        assert cleaner.clean("[图片] 看看这个") == "（图片） 看看这个"
        assert cleaner.clean("我发了个[链接]给你看看") == "我发了个（链接）给你看看"


class TestRegression:
    """Normal messages must not be affected by the two new rules."""

    def test_normal_message_untouched(self, cleaner):
        assert cleaner.clean("明天一起吃饭吗") == "明天一起吃饭吗"

    def test_image_placeholder_with_text_is_not_dropped(self, cleaner):
        # [图片] is not a share-card placeholder: keep the conversion behavior
        assert cleaner.clean("[图片] 看看这个") == "（图片） 看看这个"

    def test_normal_link_mention_untouched(self, cleaner):
        # "链接" as a plain word without brackets is normal text
        assert cleaner.clean("把那个链接发我一下") == "把那个链接发我一下"

    def test_share_card_still_dropped(self, cleaner):
        # regression: previous Bug1 fix unaffected
        assert cleaner.clean('[链接] 我校首届"璀璨时光"校园大学生超算竞赛拉开序幕！') is None

    def test_payment_scam_still_dropped(self, cleaner):
        # regression: previous Bug2 fix unaffected
        assert cleaner.clean("411 http:，覆置 zr:/CIptAOd76eR咑亓🌟支..fu宝🌟就可以帮我付款啦") is None

    def test_mid_sentence_link_placeholder_still_converted(self, cleaner):
        assert cleaner.clean("我发了个[链接]给你看看") == "我发了个（链接）给你看看"
