import sys

with open(r"d:\三创\LaRE-main\web\vue\src\views\SessionDetectionView.vue", "r", encoding="utf-8") as f:
    text = f.read()

old_str = """                  <div class="verdict-title"><i class="fa-solid fa-gavel"></i> 判决建议</div>\n                  <p v-if="isFake && detectingMsgIsOpponent">该图片由<strong>「对方」</strong>提供，AI 高置信度判定为 <strong>AIGC 生成图像</strong>，虚假证据属实，建议支持您的维权诉求。</p>\n                  <p v-else-if="isFake && !detectingMsgIsOpponent">该图片由<strong>「我方」</strong>提供，AI 判定为 <strong>AIGC 生成图像</strong>，请核实该图片来源，如有误判请申诉。</p>\n                  <p v-else>该图片未检出明显伪造痕迹，被判定为 <strong>真实图像</strong>。</p>"""

new_str = """                  <div class="verdict-title"><i class="fa-solid fa-gavel"></i> 判决建议</div>\n                  <p v-if="isFake && detectingMsgIsOpponent">该图片由<strong>「对方 <span v-if="detectingMsgSenderName">({{ detectingMsgSenderName }})</span>」</strong>提供，AI 高置信度判定为 <strong>AIGC 生成图像</strong>，虚假证据属实，建议支持您的维权诉求。</p>\n                  <p v-else-if="isFake && !detectingMsgIsOpponent">该图片由<strong>「我方 <span v-if="detectingMsgSenderName">({{ detectingMsgSenderName }})</span>」</strong>提供，AI 判定为 <strong>AIGC 生成图像</strong>，请核实该图片来源，如有误判请申诉。</p>\n                  <p v-else>该图片由<strong>「{{ detectingMsgIsOpponent ? \
对方\ : \我方\ }} <span v-if="detectingMsgSenderName">({{ detectingMsgSenderName }})</span>」</strong>提供，未检出明显伪造痕迹，被判定为 <strong>真实图像</strong>。</p>"""

if old_str in text:
    text = text.replace(old_str, new_str)
    with open(r"d:\三创\LaRE-main\web\vue\src\views\SessionDetectionView.vue";, "w", encoding="utf-8") as f:
        f.write(text)
    print("Replace success!")
else:
    print("Replace failed!")

