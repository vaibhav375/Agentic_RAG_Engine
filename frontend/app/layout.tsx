import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Self-correcting RAG",
  description: "Watch a RAG engine decide whether it is allowed to answer.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
