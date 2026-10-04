/// <reference lib="webworker" />
/** Background thread: runs the pipeline in core.ts so the page stays responsive while beans are counted. */
import { analyze, embedCrops, init } from "./core";
import type { FromWorker, ToWorker } from "./protocol";

const ctx = self as unknown as DedicatedWorkerGlobalScope;
const post = (m: FromWorker) => {
  const buf = m.type === "result" || m.type === "embedded" ? m.embeds?.buffer : undefined;
  ctx.postMessage(m, buf ? [buf as ArrayBuffer] : []);
};
let ready: Promise<void> | null = null;

ctx.onmessage = async (ev: MessageEvent<ToWorker>) => {
  const msg = ev.data;
  try {
    if (msg.type === "init") {
      ready = ready ?? init(msg.base).then(post);
      await ready;
      return;
    }
    if (!ready) throw new Error("worker not initialised");
    await ready;
    if (msg.type === "analyze") post(await analyze(msg));
    else if (msg.type === "embed") post(await embedCrops(msg));
  } catch (e) {
    post({ type: "error", id: (msg as { id?: number }).id, message: String((e as Error)?.message ?? e) });
  }
};
