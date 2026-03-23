import sys

with open(r"d:\三创\LaRE-main\web\vue\src\views\SessionDetectionView.vue", "r", encoding="utf-8") as f:
    text = f.read()

old_block = """      showForensicsReport.value = true;
      } finally {
        isGeneratingReport.value = false;
const triggerImageUpload"""

new_block = """      showForensicsReport.value = true;
    } finally {
      isGeneratingReport.value = false;
    }
  };

  // 关闭定责报告
  const closeForensicsReport = () => {
    showForensicsReport.value = false;
  };

  const handleSendText = async () => {
    if (!inputText.value.trim() || !activeSession.value) return;
    try {
      const res = await sendTextMessage(activeSession.value.id, inputText.value.trim());
      if (res.success) {
        messages.value.push(res.message);
        inputText.value = ';
        scrollToBottom();
      }
    } catch (e) {
      console.error("发送消息失败:", e);
    }
  };

  const triggerImageUpload"""

if old_block in text:
    text = text.replace(old_block, new_block)
    with open(r"d:\三创\LaRE-main\web\vue\src\views\SessionDetectionView.vue", "w", encoding="utf-8") as f:
        f.write(text)
    print("Replace success!")
else:
    print("Replace failed!")
