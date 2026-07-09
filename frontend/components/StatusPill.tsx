"use client";

import { useEffect, useState } from "react";
import { getHealth } from "@/lib/api";

type Status = "checking" | "online" | "offline";

export default function StatusPill() {
  const [status, setStatus] = useState<Status>("checking");
  const [numClasses, setNumClasses] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((h) => {
        if (cancelled) return;
        setStatus("online");
        setNumClasses(h.num_classes);
      })
      .catch(() => {
        if (!cancelled) setStatus("offline");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const dotColor =
    status === "online"
      ? "bg-leaf"
      : status === "offline"
        ? "bg-rust"
        : "bg-text-muted";

  const label =
    status === "online"
      ? `API online \u00b7 ${numClasses} classes loaded`
      : status === "offline"
        ? "API unreachable \u2014 is uvicorn running?"
        : "checking API\u2026";

  return (
    <div className="flex items-center gap-2 font-mono text-xs text-text-muted">
      <span
        className={`inline-block h-[7px] w-[7px] rounded-full ${dotColor} ${
          status === "checking" ? "pulse-dot" : ""
        }`}
      />
      {label}
    </div>
  );
}
