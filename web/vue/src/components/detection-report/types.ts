export interface ForensicsReportData {
  reportId: string;
  timestamp: string;
  platform?: string;
  targetImageUrl: string;

  sessionContext?: {
    sessionId: string;
    buyerName?: string;
    sellerName?: string;
    senderRole?: 'buyer' | 'seller' | 'unknown';
  };

  aiDetection: {
    isFake: boolean;
    confidence: number;
    probabilities?: Record<string, number>;
    heatmapBase64?: string;
  };

  vlReport?: string;
  
  riskyStatements?: Array<{
    context?: string;
    quote: string;
    riskLevel: string;
    comment: string;
  }>;
  
  ruleViolations?: Array<{
    ruleName: string;
    explanation: string;
  }>;

  actionSuggestions?: Array<{
    action: string;
    detail: string;
  }>;

  chatLog?: Array<{
    time: string;
    role: string;
    side: string;
    content: string;
  }>;
}