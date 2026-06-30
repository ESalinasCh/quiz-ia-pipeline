using System.Text.Json;
using Microsoft.Extensions.AI;
using Qdrant.Client;
using Qdrant.Client.Grpc;

namespace QuizDotnet;

partial class Program
{
    static async Task<List<SemanticChunk>> ProcessChunksAndStoreAsync(List<SemanticChunk> chunks, string courseId, string sourceId, string collectionName)
    {
        IChatClient chatClient = new OllamaChatClient(new Uri(OllamaUrl), LlmModel);
        IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator =
            new OllamaEmbeddingGenerator(new Uri(OllamaUrl), EmbeddingModel);

        // Connect to Qdrant using the official gRPC client
        Console.WriteLine("Connecting to Qdrant...");
        using var qdrantClient = new QdrantClient(QdrantHost, QdrantPort);

        // Initialize collection if it doesn't exist
        var collections = await qdrantClient.ListCollectionsAsync();
        if (!collections.Contains(collectionName))
        {
            Console.WriteLine($"Creating Qdrant collection '{collectionName}'...");
            await qdrantClient.CreateCollectionAsync(
                collectionName: collectionName,
                vectorsConfig: new VectorParams { Size = EmbeddingDim, Distance = Distance.Cosine }
            );
        }

        var processedChunks = new List<SemanticChunk>();
        var points = new List<PointStruct>();

        foreach (var chunk in chunks)
        {
            // 1. Classify relevance
            string classPrompt = $@"Eres un clasificador de contenido educativo universitario.
Clasifica el siguiente fragmento de clase grabada.

SPEAKER: {chunk.Speaker}
TEXTO: ""{chunk.Content}""

Categorías:
- ACADEMICO: contenido del tema de la materia, explicaciones, definiciones, ejemplos académicos.
- OFF_TOPIC: conversación no relacionada al tema (clima, anécdotas personales, organización, saludos).
- ERROR_ESTUDIANTIL: respuesta incorrecta de un estudiante no corregida explícitamente.
- ANECDOTA: historia ilustrativa, puede ser útil como contexto pero no es conocimiento clave.

Responde únicamente con un objeto JSON válido con este formato:
{{""categoria"": ""ACADEMICO"", ""confidence"": 0.92, ""razon"": ""explicación sobre termodinámica""}}";

            var classOptions = new ChatOptions
            {
                ResponseFormat = ChatResponseFormat.ForJsonSchema(typeof(ClassificationResult))
            };

            string category = "ACADEMICO";
            double confidence = 0.8;
            try
            {
                var classResponse = await chatClient.GetResponseAsync(classPrompt, classOptions);
                var classResult = JsonSerializer.Deserialize<ClassificationResult>(classResponse.Text);
                if (classResult != null)
                {
                    category = classResult.Categoria.ToUpperInvariant();
                    confidence = classResult.Confidence;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Classification error for chunk {chunk.ChunkIndex}: {ex.Message}");
            }

            chunk.Category = category;
            chunk.ConfidenceScore = confidence;
            chunk.QdrantVectorId = Guid.NewGuid();

            // 2. Compute embedding vector (guarded: bge-m3 can emit a NaN vector Ollama cannot serialize).
            float[] vector;
            try
            {
                var embRes = await embeddingGenerator.GenerateAsync(new[] { chunk.Content });
                vector = embRes[0].Vector.ToArray();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Skipping chunk {chunk.ChunkIndex} that failed to embed: {ex.Message}");
                continue;
            }

            // 3. Stage point for a single batched upsert
            var point = new PointStruct
            {
                Id = new PointId { Uuid = chunk.QdrantVectorId.ToString() },
                Vectors = vector
            };
            point.Payload.Add("course_id", courseId); // course_id belongs to the ClassSource
            point.Payload.Add("source_id", sourceId); // The equivalent would be document_id (FK to ClassSource)
            point.Payload.Add("content", chunk.Content);
            point.Payload.Add("speaker", chunk.Speaker); // Not used on real project, this version doesn't divide speakers
            point.Payload.Add("category", chunk.Category); // Not used on real project, only academic are stored
            point.Payload.Add("confidence_score", chunk.ConfidenceScore); // Not used on real project, only chunks with high score are stored
            point.Payload.Add("ts_start", chunk.TsStart); // Not used on real project
            point.Payload.Add("ts_end", chunk.TsEnd); // Not used on real project
            points.Add(point);

            Console.WriteLine($"Processed chunk {chunk.ChunkIndex} (ID: {chunk.QdrantVectorId}, Category: {chunk.Category}).");
            processedChunks.Add(chunk);
        }

        // 4. Index all chunks in Qdrant in a single batched upsert.
        if (points.Count > 0)
        {
            await qdrantClient.UpsertAsync(collectionName, points);
            Console.WriteLine($"Upserted {points.Count} points into Qdrant collection '{collectionName}'.");
        }

        return processedChunks;
    }
}
