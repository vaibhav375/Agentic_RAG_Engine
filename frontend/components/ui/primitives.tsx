"use client";

import { cn } from "@/lib/utils";
import * as React from "react";

export function Card({ className, ...p }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("rounded-xl border border-edge bg-panel", className)} {...p} />;
}

export function CardHeader({ className, ...p }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("border-b border-edge px-4 py-3 text-sm font-semibold", className)} {...p} />;
}

export function CardBody({ className, ...p }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("p-4", className)} {...p} />;
}

export function Button({
  className,
  variant = "solid",
  ...p
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "solid" | "ghost" }) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition disabled:opacity-40",
        variant === "solid" && "bg-accent text-black hover:brightness-110",
        variant === "ghost" && "border border-edge text-white hover:bg-edge/40",
        className,
      )}
      {...p}
    />
  );
}

export function Switch({
  checked,
  onChange,
  label,
  hint,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  hint?: string;
}) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-3 py-1.5">
      <span className="text-sm">
        {label}
        {hint && <span className="ml-1 text-xs text-muted">· {hint}</span>}
      </span>
      <button
        type="button"
        onClick={() => onChange(!checked)}
        className={cn(
          "relative h-5 w-9 shrink-0 rounded-full transition",
          checked ? "bg-accent" : "bg-edge",
        )}
        aria-pressed={checked}
      >
        <span
          className={cn(
            "absolute top-0.5 h-4 w-4 rounded-full bg-black transition",
            checked ? "left-[18px]" : "left-0.5",
          )}
        />
      </button>
    </label>
  );
}

export function Badge({
  children,
  tone = "default",
}: {
  children: React.ReactNode;
  tone?: "default" | "good" | "warn" | "bad" | "info";
}) {
  const tones: Record<string, string> = {
    default: "border-edge text-muted",
    good: "border-accent/50 text-accent",
    warn: "border-warn/50 text-warn",
    bad: "border-danger/50 text-danger",
    info: "border-accent2/60 text-[#c9a0e8]",
  };
  return (
    <span className={cn("rounded-full border px-2.5 py-0.5 text-xs font-medium", tones[tone])}>
      {children}
    </span>
  );
}
