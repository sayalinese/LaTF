import sys

with open(r"d:\三创\LaRE-main\web\vue\src\components\detection-report\ReportDocument.vue", "r", encoding="utf-8") as f:
    content = f.read()

old_sec_4 = """      <!-- Section 4: Multimodal Analysis -->
      <section class="report-section" v-if="data.vlReport">
        <h2 class="section-title"><span class="bg-icon">4</span> 多模态综合分析</h2>
        <div class="section-body">
          <div class="markdown-body vl-report-content" v-html="renderMd(data.vlReport)"></div>
        </div>
      </section>"""

new_sections = """      <!-- Section 4: Multimodal Analysis -->
      <section class="report-section" v-if="data.vlReport">
        <h2 class="section-title"><span class="bg-icon">4</span> 视觉破绽解析 (多模态)</h2>
        <div class="section-body">
          <div class="markdown-body vl-report-content" v-html="renderMd(data.vlReport)"></div>
        </div>
      </section>

      <!-- Section 5: Risky Statements -->
      <section class="report-section" v-if="data.riskyStatements && data.riskyStatements.length > 0">
        <h2 class="section-title"><span class="bg-icon">5</span> 对方违规风险言论取证</h2>
        <div class="section-body">
          <table class="risk-table">
            <thead>
              <tr>
                <th style="width: 100px;">风险等级</th>
                <th>原话引用</th>
                <th>AI分析点评</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(item, idx) in data.riskyStatements" :key="idx">
                <td :class="\
risk-badge
\ + item.riskLevel.toLowerCase()">{{ item.riskLevel }}</td>
                <td class="quote">{{ item.quote }}</td>
                <td class="comment">{{ item.comment }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <!-- Section 6: Rule Violations -->
      <section class="report-section" v-if="data.ruleViolations && data.ruleViolations.length > 0">
        <h2 class="section-title"><span class="bg-icon">6</span> 平台风控违规判定</h2>
        <div class="section-body">
          <ul class="rule-list">
            <li v-for="(rule, idx) in data.ruleViolations" :key="idx" class="rule-item">
              <div class="rule-header"><i class="fa-solid fa-gavel"></i> <strong>{{ rule.ruleName }}</strong></div>
              <div class="rule-body">{{ rule.explanation }}</div>
            </li>
          </ul>
        </div>
      </section>"""

content = content.replace(old_sec_4, new_sections)

# Also add some CSS for the new sections
style_to_add = """
.risk-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 10px;
}
.risk-table th, .risk-table td {
  border: 1px solid #e2e8f0;
  padding: 8px 12px;
  text-align: left;
}
.risk-table th {
  background: #f8fafc;
  font-weight: 600;
  color: #334155;
}
.risk-badge {
  font-weight: bold;
  text-align: center;
  border-radius: 4px;
}
.risk-badge.high { color: #dc2626; background: #fee2e2; }
.risk-badge.medium { color: #f59e0b; background: #fef3c7; }
.risk-badge.low { color: #16a34a; background: #dcfce7; }
.rule-list {
  list-style: none;
  padding: 0;
  margin-top: 10px;
}
.rule-item {
  background: #f1f5f9;
  border-left: 4px solid #3b82f6;
  padding: 12px;
  margin-bottom: 10px;
  border-radius: 0 6px 6px 0;
}
.rule-header {
  color: #1e3a8a;
  margin-bottom: 6px;
  font-size: 15px;
}
.rule-body {
  color: #475569;
  font-size: 14px;
}
</style>
"""
if "</style>" in content and ".risk-table" not in content:
    content = content.replace("</style>", style_to_add)

with open(r"d:\三创\LaRE-main\web\vue\src\components\detection-report\ReportDocument.vue", "w", encoding="utf-8") as f:
    f.write(content)

