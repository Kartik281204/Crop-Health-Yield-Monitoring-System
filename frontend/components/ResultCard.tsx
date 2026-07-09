import { PredictionResponse, formatLabel, isHealthyLabel } from "@/lib/api";

export default function ResultCard({ result }: { result: PredictionResponse }) {
  const healthy = isHealthyLabel(result.predicted_class);

  return (
    <div className="rounded-lg border border-border bg-surface p-5">
      <div className="mb-3.5 font-mono text-[11px] uppercase tracking-[0.1em] text-text-muted">
        prediction
      </div>

      <div className="mb-5 flex items-baseline justify-between">
        <span
          className={`text-xl font-semibold ${healthy ? "text-leaf" : "text-rust"}`}
        >
          {formatLabel(result.predicted_class)}
        </span>
        <span className="font-mono text-sm text-text-muted">
          {(result.confidence * 100).toFixed(1)}%
        </span>
      </div>

      <div className="mb-3 font-mono text-[11px] uppercase tracking-[0.1em] text-text-muted">
        top 3
      </div>
      <div className="space-y-2.5">
        {result.top_3.map((item) => {
          const h = isHealthyLabel(item.label);
          const pct = (item.confidence * 100).toFixed(1);
          return (
            <div
              key={item.label}
              className="grid grid-cols-[1fr_46px] items-center gap-2.5 font-mono text-[12.5px]"
            >
              <div className="flex flex-col gap-1">
                <span className="text-text">{formatLabel(item.label)}</span>
                <div className="h-[7px] overflow-hidden rounded-full bg-surface-2">
                  <div
                    className={`h-full rounded-full transition-[width] duration-500 ease-out ${
                      h ? "bg-leaf-dim" : "bg-rust-dim"
                    }`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
              <span className="text-right text-text-muted">{pct}%</span>
            </div>
          );
        })}
      </div>

      {!healthy && (
        <p className="mt-4 border-t border-border pt-3 text-[12.5px] leading-relaxed text-text-muted">
          Flagged as diseased. This model is a portfolio project, not a
          diagnostic tool &mdash; confirm with a local agricultural extension
          service before acting on it.
        </p>
      )}
    </div>
  );
}
