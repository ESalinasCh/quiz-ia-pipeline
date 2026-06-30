using System.Numerics.Tensors;
using Microsoft.Extensions.AI;

namespace QuizDotnet;

partial class Program
{
    // * SplitIntoSentences divide the transcript in sentences. Each sentence is made of 2 versions: a string of 15 words or a string that ends with a punctuation.
    static List<Sentence> SplitIntoSentences(List<TranscriptSegment> segments)
    {
        var sentences = new List<Sentence>();
        var currentWords = new List<string>();
        double? startTime = null;
        var speakers = new List<string>();

        foreach (var seg in segments)
        {
            string text = seg.Text.Trim();
            if (string.IsNullOrWhiteSpace(text)) continue;

            if (startTime == null) startTime = seg.Start;
            currentWords.Add(text);
            speakers.Add(seg.Speaker);

            bool endsWithPunc = text.EndsWith('.') || text.EndsWith('?') || text.EndsWith('!');
            int wordCount = currentWords.Sum(w => w.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length);

            if (endsWithPunc || wordCount >= 15)
            {
                string dominantSpeaker = speakers.GroupBy(s => s)
                                                 .OrderByDescending(g => g.Count())
                                                 .First().Key;
                sentences.Add(new Sentence(
                    string.Join(" ", currentWords),
                    startTime.Value,
                    seg.End,
                    dominantSpeaker
                ));
                currentWords.Clear();
                startTime = null;
                speakers.Clear();
            }
        }

        if (currentWords.Count > 0 && segments.Count > 0)
        {
            string dominantSpeaker = speakers.GroupBy(s => s)
                                             .OrderByDescending(g => g.Count())
                                             .FirstOrDefault()?.Key ?? "SPEAKER_0";
            sentences.Add(new Sentence(
                string.Join(" ", currentWords),
                startTime ?? segments[0].Start,
                segments[^1].End,
                dominantSpeaker
            ));
        }

        return sentences;
    }

    static async Task<List<SemanticChunk>> GenerateSemanticChunksAsync(List<Sentence> sentences)
    {
        if (sentences.Count == 0) return new List<SemanticChunk>();

        Console.WriteLine($"Computing sentence embeddings using Ollama ({EmbeddingModel})...");
        IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator =
            new OllamaEmbeddingGenerator(new Uri(OllamaUrl), EmbeddingModel);

        // Embed sentences in batches so the embedding model is not called for each sentence with per-item fallback. bge-m3 returns a NaN vector for some
        // degenerate inputs, which Ollama cannot JSON-encode and which would kill an all-in-one batch call. 
        var (sentencesEmbedded, embeddings) = await EmbedSentencesAsync(embeddingGenerator, sentences);
        if (sentencesEmbedded.Count == 0) return new List<SemanticChunk>();
        sentences = sentencesEmbedded; // keep sentences and embeddings index-aligned after any drops

        var similarities = new List<float>();
        for (int i = 0; i < embeddings.Count - 1; i++)
        {
            float sim = TensorPrimitives.CosineSimilarity(embeddings[i].Vector.Span, embeddings[i + 1].Vector.Span);
            similarities.Add(sim);
        }

        var chunks = new List<SemanticChunk>();
        var currentSentences = new List<Sentence> { sentences[0] };
        int currentWordCount = sentences[0].Text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;
        int chunkIndex = 0;

        for (int i = 0; i < similarities.Count; i++)
        {
            float sim = similarities[i];
            var nextSentence = sentences[i + 1];
            int nextWordCount = nextSentence.Text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;

            bool shouldSplit = false;
            // This fisrt if evaluates if the contiguous sentences aren't similar and are at least 80 words long. 
            if (sim < ChunkSimilarityThreshold)
            {
                if (currentWordCount >= MinChunkWords)
                {
                    shouldSplit = true;
                }
            }

            // This second if evaluates if the current sentences and the next one adds up more than 400 words.
            if (currentWordCount + nextWordCount > MaxChunkWords)
            {
                shouldSplit = true;
            }

            // If any of the previous conditions are met, a chunk is created
            if (shouldSplit)
            {
                chunks.Add(BuildChunk(currentSentences, chunkIndex++));
                currentSentences = new List<Sentence> { nextSentence };
                currentWordCount = nextWordCount;
            }
            else
            {
                currentSentences.Add(nextSentence);
                currentWordCount += nextWordCount;
            }
        }

        if (currentSentences.Count > 0)
        {
            chunks.Add(BuildChunk(currentSentences, chunkIndex));
        }

        return chunks;
    }

    // * Embeds sentences in batches with a per-item fallback. Returns the surviving sentences and their embeddings, kept strictly index-aligned (degenerate inputs that fail to embed are dropped from both).
    static async Task<(List<Sentence> Sentences, List<Embedding<float>> Embeddings)> EmbedSentencesAsync(
        IEmbeddingGenerator<string, Embedding<float>> generator, List<Sentence> sentences)
    {
        const int batchSize = 32; // 32 sentences
        var keptSentences = new List<Sentence>();
        var embeddings = new List<Embedding<float>>();

        // Drop empty/whitespace sentences up front: bge-m3 yields NaN vectors for them, which Ollama
        // cannot serialize ("json: unsupported value: NaN").
        var clean = sentences.Where(s => !string.IsNullOrWhiteSpace(s.Text)).ToList();

        for (int i = 0; i < clean.Count; i += batchSize)
        {
            var batch = clean.GetRange(i, Math.Min(batchSize, clean.Count - i));
            try
            {
                var res = await generator.GenerateAsync(batch.Select(s => s.Text).ToList());
                for (int j = 0; j < batch.Count; j++)
                {
                    keptSentences.Add(batch[j]);
                    embeddings.Add(res[j]);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Batch embedding failed ({ex.Message}); retrying items individually...");
                foreach (var s in batch)
                {
                    try
                    {
                        var single = await generator.GenerateAsync(new[] { s.Text });
                        keptSentences.Add(s);
                        embeddings.Add(single[0]);
                    }
                    catch (Exception exItem)
                    {
                        Console.WriteLine($"Skipping sentence that failed to embed: {exItem.Message}");
                    }
                }
            }
        }

        return (keptSentences, embeddings);
    }

    static SemanticChunk BuildChunk(List<Sentence> chunkSents, int index)
    {
        string content = string.Join(" ", chunkSents.Select(s => s.Text));
        string dominantSpeaker = chunkSents.GroupBy(s => s.Speaker)
                                           .OrderByDescending(g => g.Count())
                                           .First().Key;
        return new SemanticChunk(index, chunkSents[0].Start, chunkSents[^1].End, dominantSpeaker, content);
    }
}
