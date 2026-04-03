import os
import sys
import json
from pathlib import Path

def read_top_k_chunks(k=3, answer_dir="answer"):
    chunks = []
    for i in range(1, k+1):
        p = Path(answer_dir) / f"{i}.txt"
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        # assume chunk text is after the first blank line
        parts = text.split("\n\n", 1)
        metadata = parts[0] if parts else ""
        body = parts[1].strip() if len(parts) > 1 else ""
        chunks.append({"file": str(p), "meta": metadata, "text": body})
    return chunks


def build_prompt(query, chunks):
    prompt = []
    prompt.append("You are an expert assistant that answers questions based on provided document contexts.")
    prompt.append("Given the user query and the document snippets below, provide a concise, factual, and referenced answer. If the answer is not contained in the snippets, say you don't know and suggest how to find the answer.")
    prompt.append("")
    prompt.append(f"User query: {query}")
    prompt.append("")
    prompt.append("Contexts:")
    for i, c in enumerate(chunks, 1):
        prompt.append(f"[{i}] Source: {c['file']}")
        # show a short metadata line
        meta_line = c['meta'].replace('\n', ' | ')
        prompt.append(f"{meta_line}")
        prompt.append(c['text'])
        prompt.append("")
    prompt.append("Answer the query using the contexts above. Quote small excerpts if needed and cite the context number(s) in square brackets, e.g., [1]. Keep the answer concise (3-5 sentences) and include a short confidence statement.")
    return "\n".join(prompt)


def call_openai_chat(prompt, model="gpt-3.5-turbo", temperature=0.0):
    try:
        import openai
    except Exception as e:
        raise RuntimeError("OpenAI package is not installed. Install with `pip install openai`") from e
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    openai.api_key = api_key
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=512,
    )
    return resp["choices"][0]["message"]["content"].strip()


def call_google_generative(prompt, model=None, api_key=None, temperature=0.0):
    """
    Call Google's Generative Language API. Defaults to the Gemini preview model
    `gemini-3-flash-preview` unless overridden via the `model` arg or `GOOGLE_MODEL` env.
    Prefers the `google-genai` SDK if installed, otherwise falls back to REST (tries v1beta2 and v1).
    """
    try:
        import requests
    except Exception as e:
        raise RuntimeError("requests not installed. Install with `pip install requests`") from e

    if model is None:
        model = os.environ.get("GOOGLE_MODEL", "gemini-3-flash-preview")

    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        # fallback: look for local file
        if not api_key and Path("Abhi_Gemini_API.txt").exists():
            api_key = Path("Abhi_Gemini_API.txt").read_text(encoding="utf-8").strip()

    if not api_key:
        raise RuntimeError("Google API key not found. Set GOOGLE_API_KEY or create 'Abhi_Gemini_API.txt' with the key.")

    # Prefer the official google-genai SDK if it's available (installed via `google-genai`)
    try:
        # Try both common SDK import names
        try:
            import google.generativeai as genai_sdk
        except Exception:
            import google.genai as genai_sdk

        # If SDK provides a Client class (google.genai), use it
        if hasattr(genai_sdk, "Client"):
            client = genai_sdk.Client(api_key=api_key)
            # The client expects raw prompt text in `contents` and config dict for params
            cfg = {"max_output_tokens": 512, "temperature": float(temperature)}
            resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
            # resp may be a SDK object with .candidates
            candidates = None
            try:
                candidates = list(resp.candidates)
            except Exception:
                try:
                    candidates = resp.get("candidates", [])
                except Exception:
                    candidates = []

            if candidates and len(candidates) > 0:
                cand = candidates[0]
                # SDK candidate may have .content.parts[0].text
                try:
                    return cand.content.parts[0].text.strip()
                except Exception:
                    try:
                        return cand.get("content", {}).get("parts", [])[0].get("text", "").strip()
                    except Exception:
                        return str(cand).strip()

        # Fallback: some older SDKs expose a generate_text convenience function
        if hasattr(genai_sdk, "generate_text"):
            sdk_model = model if model.startswith("models/") else f"models/{model}"
            resp = genai_sdk.generate_text(model=sdk_model, temperature=float(temperature), max_output_tokens=512, prompt=prompt)
            if isinstance(resp, dict):
                if "candidates" in resp and len(resp["candidates"]) > 0:
                    return resp["candidates"][0].get("content", resp["candidates"][0].get("output", "")).strip()
                return resp.get("output", {}).get("text", "").strip()
            return getattr(resp, "text", str(resp)).strip()
    except Exception:
        # Fall back to REST call if SDK not available or fails
        pass

    # Try known API base versions in order. Some accounts / regions may support v1beta2 or v1.
    for api_version in ("v1beta2", "v1"):
        url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:generate?key={api_key}"
        headers = {"Content-Type": "application/json; charset=utf-8"}
        payload = {
            "prompt": {"text": prompt},
            "temperature": float(temperature),
            "maxOutputTokens": 512
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            j = resp.json()
            if "candidates" in j and len(j["candidates"]) > 0:
                cand = j["candidates"][0]
                if isinstance(cand, dict):
                    return cand.get("output", cand.get("content", "")).strip()
                return str(cand).strip()
            return j.get("output", {}).get("text", "").strip()
        if resp.status_code == 404:
            # try next api_version
            continue
        raise RuntimeError(f"Google Generative API error: {resp.status_code} {resp.text}")

    raise RuntimeError("Google Generative API endpoint not found (tried v1beta2 and v1). Check API access and model name.")


def call_hf_flan(prompt, model_name="google/flan-t5-small", device=-1):
    # device: -1 -> CPU, >=0 GPU index
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
    except Exception as e:
        raise RuntimeError("transformers/torch not installed. Install with `pip install transformers torch`") from e

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if device >= 0 and torch.cuda.is_available():
        model = model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    if device >= 0 and torch.cuda.is_available():
        inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    ans = tokenizer.decode(out[0], skip_special_tokens=True)
    return ans.strip()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 llm_answer.py \"your query here\"")
        sys.exit(1)
    query = sys.argv[1]
    chunks = read_top_k_chunks(k=3)
    if not chunks:
        print("No top chunk files found in `answer/`. Run the pipeline first to populate answer/1.txt..3.txt")
        sys.exit(1)
    prompt = build_prompt(query, chunks)

    # Try OpenAI if key is present, otherwise HF fallback
    answer = None
    try:
        # priority: GOOGLE_API_KEY -> OPENAI_API_KEY -> HF fallback
        if os.environ.get("GOOGLE_API_KEY") or Path("Abhi_Gemini_API.txt").exists():
            print("Using Google Generative API (AI Studio / text-bison) for generation...")
            answer = call_google_generative(prompt)
        elif os.environ.get("OPENAI_API_KEY"):
            print("Using OpenAI API for generation...")
            answer = call_openai_chat(prompt)
        else:
            print("No cloud API key set — using HuggingFace Flan-T5 fallback (cpu)...")
            answer = call_hf_flan(prompt)
    except Exception as e:
        print("LLM call failed:", e)
        sys.exit(1)

    out_path = Path("answer") / "response.txt"
    out_path.write_text(answer, encoding="utf-8")
    print(f"Answer written to {out_path}")
    print("\n--- ANSWER ---\n")
    print(answer)


if __name__ == "__main__":
    main()


def generate_answer(query, k=3):
    """Generate an answer for `query` using top-k chunks from the `answer/` folder.
    Returns the answer string (also writes to `answer/response.txt`)."""
    chunks = read_top_k_chunks(k=k)
    if not chunks:
        raise RuntimeError("No chunks found in answer/ — run pipeline first")
    prompt = build_prompt(query, chunks)

    # priority: GOOGLE_API_KEY -> OPENAI_API_KEY -> HF fallback
    # If Google call fails, fall back to OpenAI or HF instead of raising.
    answer = None
    if os.environ.get("GOOGLE_API_KEY") or Path("Abhi_Gemini_API.txt").exists():
        try:
            answer = call_google_generative(prompt)
        except Exception as e:
            print(f"Google Generative API call failed, falling back: {e}")

    if answer is None and os.environ.get("OPENAI_API_KEY"):
        try:
            answer = call_openai_chat(prompt)
        except Exception as e:
            print(f"OpenAI call failed, falling back: {e}")

    if answer is None:
        answer = call_hf_flan(prompt)

    out_path = Path("answer") / "response.txt"
    out_path.write_text(answer, encoding="utf-8")
    return answer
