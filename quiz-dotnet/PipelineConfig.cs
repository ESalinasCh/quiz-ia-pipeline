namespace QuizDotnet;

// Central configuration for the pipeline. Single source of truth for models, endpoints, and tuning.
partial class Program
{
    // Ollama models used across the pipeline.
    const string LlmModel = "qwen2.5:7b-instruct";          // classification, summarization, quiz generation
    const string JudgeModel = "llama3.1:8b-instruct-q4_K_M"; // LLM-as-a-judge (different family to reduce self-bias)
    const string EmbeddingModel = "bge-m3";                  // multilingual embeddings (1024-dim)
    const int EmbeddingDim = 1024;

    // Service endpoints.
    const string OllamaUrl = "http://127.0.0.1:11434";
    const string QdrantHost = "127.0.0.1";
    const int QdrantPort = 6334; // gRPC default

    // Semantic chunking tuning. Higher MinChunkWords + lower similarity threshold => fewer, larger chunks
    // (fewer classification LLM calls, more context per question).
    const float ChunkSimilarityThreshold = 0.45f; // split only on clearer topic shifts (was 0.5)
    const int MinChunkWords = 80;                  // don't split until a chunk has this many words (was 30)
    const int MaxChunkWords = 400;                 // hard cap before a forced split (was 350)

    // Adaptive quiz size: ~1 question per QuestionsPerChunkDivisor academic chunks, clamped to [Min, Max].
    const double QuestionsPerChunkDivisor = 5.0;   // higher => fewer questions
    const int MinQuestions = 5;
    const int MaxQuestions = 25;

    // Fraction of questions generated as true/false (rest are multiple choice). 0 disables T/F.
    const double TrueFalseRatio = 0.3;
}
