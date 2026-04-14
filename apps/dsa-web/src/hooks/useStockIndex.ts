/**
 * useStockIndex Hook
 *
 * Manage stock index loading and state
 */

import { useState, useEffect } from 'react';
import type { StockIndexItem } from '../types/stockIndex';
import { loadStockIndex } from '../utils/stockIndexLoader';
import type { IndexLoadResult } from '../utils/stockIndexLoader';

export interface UseStockIndexResult {
  /** Stock index data */
  index: StockIndexItem[];
  /** Is loading */
  loading: boolean;
  /** Load error */
  error: Error | null;
  /** Whether fallback mode is used */
  fallback: boolean;
  /** Is loaded */
  loaded: boolean;
}

/**
 * Stock index loading Hook
 *
 * @returns Index state and data
 */
export function useStockIndex(): UseStockIndexResult {
  const [index, setIndex] = useState<StockIndexItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [fallback, setFallback] = useState(false);

  useEffect(() => {
    let mounted = true;

    async function load() {
      setLoading(true);
      setError(null);

      const result: IndexLoadResult = await loadStockIndex();

      if (mounted) {
        setIndex(result.data);
        // 索引加载成功但为空时同样触发 fallback，避免出现"永不弹出建议"的静默失效
        setFallback(result.fallback || result.data.length === 0);
        if (result.error) {
          setError(result.error);
        }
        setLoading(false);
      }
    }

    load();

    return () => {
      mounted = false;
    };
  }, []);

  return {
    index,
    loading,
    error,
    fallback,  // Whether fallback
    loaded: !loading,
  };
}

/**
 * Get default exported Hook
 */
export default useStockIndex;
