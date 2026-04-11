/**
 * Query Normalization Utilities
 *
 * Normalize user input for stock search
 */

/**
 * Normalize query string
 *
 * @param query - Raw query string
 * @returns Normalized query string
 */
export function normalizeQuery(query: string): string {
  if (!query) return '';

  // Trim whitespace and convert to lowercase
  let normalized = query.trim().toLowerCase();

  // Replace full-width characters with half-width
  normalized = normalized.replace(/[\uff01-\uff5e]/g, (ch) =>
    String.fromCharCode(ch.charCodeAt(0) - 0xfee0)
  );

  // Remove dot suffix for market codes (e.g., "600519.sh" → "600519")
  const marketSuffix = extractMarketSuffix(normalized);
  if (marketSuffix) {
    normalized = normalized.replace(/\.[a-z]+$/, '');
  }

  return normalized;
}

/**
 * Check if a character is Chinese
 */
export function isChineseChar(ch: string): boolean {
  const code = ch.charCodeAt(0);
  return code >= 0x4e00 && code <= 0x9fff;
}

/**
 * Check if string contains Chinese characters
 */
export function containsChinese(str: string): boolean {
  return /[\u4e00-\u9fff]/.test(str);
}

/**
 * Extract market suffix from query
 *
 * @param query - Normalized query string
 * @returns Market suffix or null
 */
export function extractMarketSuffix(query: string): string | null {
  const match = query.match(/\.([a-z]+)$/);
  if (!match) return null;

  const suffix = match[1].toUpperCase();
  const validSuffixes = ['SH', 'SZ', 'BJ', 'HK'];
  return validSuffixes.includes(suffix) ? suffix : null;
}
