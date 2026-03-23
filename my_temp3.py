import re

with open(r'd:\三创\LaRE-main\web\vue\src\views\SessionDetectionView.vue', 'r', encoding='utf-8') as f:
    content = f.read()

old_btn_pattern = r'<button class="export-report-btn".*?</button>'

new_btn = '''<button class="export-report-btn" :disabled="isGeneratingReport" @click="openForensicsReport" style="margin-top: 15px; width: 100%; border: none; background: #3b82f6; color: white; padding: 10px; border-radius: 6px; cursor: pointer; font-size: 14px; display: flex; align-items: center; justify-content: center; gap: 8px; transition: background 0.2s;">
                      <template v-if="!isGeneratingReport">
                        <i class="fa-solid fa-file-pdf"></i> 生成完整定责报告
                      </template>
                      <template v-else>
                        <i class="fa-solid fa-spinner fa-spin"></i> 正在分析语料及规则...
                      </template>
                  </button>'''

content = re.sub(old_btn_pattern, new_btn, content, flags=re.DOTALL)

if ".export-report-btn:disabled" not in content:
    content = content.replace('</style>', '.export-report-btn:disabled { background: #94a3b8 !important; cursor: not-allowed !important; }\n</style>')

with open(r'd:\三创\LaRE-main\web\vue\src\views\SessionDetectionView.vue', 'w', encoding='utf-8') as f:
    f.write(content)

print('Success')
