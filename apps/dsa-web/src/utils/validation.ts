interface ValidationResult {
  valid: boolean;
  message?: string;
  normalized: string;
}

// 兼容 A/H/美股常见代码格式的基础校验
export const validateStockCode = (value: string): ValidationResult => {
  const normalized = value.trim().toUpperCase();

  if (!normalized) {
    return { valid: false, message: '请输入股票代码', normalized };
  }

  const patterns = [
    /^\d{6}$/, // A 股 6 位数字
    /^(SH|SZ)\d{6}$/, // A 股带交易所前缀
    /^\d{5}$/, // 港股 5 位数字（无前缀）
    /^HK\d{1,5}$/, // 港股 HK 前缀格式，如 HK00700、HK01810、HK1810
    /^\d{1,5}\.HK$/, // 港股 .HK 后缀格式，如 00700.HK、1810.HK
    /^[A-Z]{1,6}(\.[A-Z]{1,2})?$/, // 美股常见 Ticker
  ];

  const valid = patterns.some((regex) => regex.test(normalized));

  return {
    valid,
    message: valid ? undefined : '股票代码格式不正确',
    normalized,
  };
};

/**
 * 判断输入是否明显不是有效的股票查询（过短、纯标点/空白等）
 */
export const isObviouslyInvalidStockQuery = (query: string): boolean => {
  const trimmed = query.trim();
  if (trimmed.length === 0) return true;
  // 纯标点或空白
  if (/^[\s\p{P}\p{S}]+$/u.test(trimmed)) return true;
  // 单个非字母数字非中文字符
  if (trimmed.length === 1 && !/[\da-zA-Z\u4e00-\u9fff]/.test(trimmed)) return true;
  return false;
};

/**
 * 判断输入是否看起来像股票代码（纯数字、带交易所前缀/后缀、英文 Ticker 等）
 */
export const looksLikeStockCode = (value: string): boolean => {
  const normalized = value.trim().toUpperCase();
  if (!normalized) return false;
  const codePatterns = [
    /^\d{4,6}$/, // 纯数字 4-6 位
    /^(SH|SZ|BJ)\d{6}$/, // A 股带前缀
    /^HK\d{1,5}$/, // 港股 HK 前缀
    /^\d{1,5}\.HK$/, // 港股 .HK 后缀
    /^[A-Z]{1,6}$/, // 美股 Ticker
    /^[A-Z]{1,6}\.[A-Z]{1,2}$/, // 美股带后缀
  ];
  return codePatterns.some((p) => p.test(normalized));
};
