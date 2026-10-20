"use client";
import Script from "next/script";

export default function DeployPage() {
  return (
    <div className="max-w-6xl mx-auto px-6 py-8 flex flex-col items-center justify-center min-h-[70vh] text-center">
      <div style={{ width: 240, height: 240 }}>
        {/* @ts-ignore */}
        <lottie-player
          src="/cooking.json"
          background="transparent"
          speed="1"
          style={{ width: "100%", height: "100%" }}
          loop
          autoplay
        />
      </div>
      <h1 className="font-display font-bold text-3xl text-text mt-6 mb-2">AWS SageMaker Deployment is Cooking</h1>
      <p className="text-sm text-subtle font-body max-w-md">
        We are currently cooking up the integration for SageMaker container deployment. Check back soon!
      </p>
      <Script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js" strategy="lazyOnload" />
    </div>
  );
}
