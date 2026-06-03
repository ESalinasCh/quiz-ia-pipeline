using System.Text.Json.Serialization;

namespace QuizDotnet;

public record TranscriptSegment(string Text, double Start, double End, string Speaker);

public record Sentence(string Text, double Start, double End, string Speaker);

public record SemanticChunk(int ChunkIndex, double TsStart, double TsEnd, string Speaker, string Content)
{
    public string Category { get; set; } = "ACADEMICO";
    public double ConfidenceScore { get; set; } = 0.8;
    public Guid QdrantVectorId { get; set; }
}

public class ClassificationResult
{
    [JsonPropertyName("categoria")]
    public string Categoria { get; set; } = "ACADEMICO";

    [JsonPropertyName("confidence")]
    public double Confidence { get; set; } = 0.8;

    [JsonPropertyName("razon")]
    public string Razon { get; set; } = "";
}

public class QuizQuestion
{
    [JsonPropertyName("question")]
    public string Question { get; set; } = "";

    [JsonPropertyName("question_type")]
    public string QuestionType { get; set; } = "multiple_choice"; // "multiple_choice" | "true_false"

    [JsonPropertyName("options")]
    public Dictionary<string, string> Options { get; set; } = new();

    [JsonPropertyName("correct_option")]
    public string CorrectOption { get; set; } = "";

    [JsonPropertyName("bloom_level")]
    public string BloomLevel { get; set; } = "";

    [JsonPropertyName("justification")]
    public string Justification { get; set; } = "";
}

public class ValidationResult
{
    [JsonPropertyName("score")]
    public double Score { get; set; } = 0.0;

    [JsonPropertyName("reason")]
    public string Reason { get; set; } = "";
}
