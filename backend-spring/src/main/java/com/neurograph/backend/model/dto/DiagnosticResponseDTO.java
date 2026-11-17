package com.neurograph.backend.model.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * Data Transfer Object returned to clinicians after GAT multimodal or Facial CNN inference.
 * 
 * Why: Bundles diagnostic probabilities, classification results, top neural saliency pathways,
 * facial landmark topologies, and Three.js / Canvas visualization payloads.
 * 
 * Configured with @JsonIgnoreProperties(ignoreUnknown = true) to prevent Jackson deserialization
 * failures when FastAPI returns model-specific or extended metadata fields.
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
@ToString(exclude = {"connectomeGraph", "imageThumbnailBase64"})
@JsonIgnoreProperties(ignoreUnknown = true)
public class DiagnosticResponseDTO {

    @JsonProperty("id")
    private Long id;

    @JsonProperty("subject_id")
    private String subjectId;

    @JsonProperty("age")
    private Double age;

    @JsonProperty("sex")
    private Integer sex;

    @JsonProperty("full_scale_iq")
    private Double fullScaleIq;

    @JsonProperty("site_id")
    private String siteId;

    @JsonProperty("predicted_class")
    private Integer predictedClass;

    @JsonProperty("predicted_label")
    private String predictedLabel;

    @JsonProperty("asd_probability")
    private Double asdProbability;

    @JsonProperty("control_probability")
    private Double controlProbability;

    @JsonProperty("confidence_percentage")
    private Double confidencePercentage;

    // Direct mappings for FastAPI ImagePredictionResponse schema
    @JsonProperty("confidence")
    private Double confidence;

    @JsonProperty("probability_asd")
    private Double probabilityAsd;

    @JsonProperty("probability_typical")
    private Double probabilityTypical;

    @JsonProperty("inference_time_ms")
    private Double inferenceTimeMs;

    @JsonProperty("inference_latency_ms")
    private Double inferenceLatencyMs;

    @JsonProperty("model_used")
    private String modelUsed;

    @JsonProperty("top_pathways")
    private List<SaliencyPathwayDTO> topPathways;

    @JsonProperty("top_biomarker_rois")
    private List<BiomarkerRoiDTO> topBiomarkerRois;

    @JsonProperty("network_attribution")
    private Map<String, Double> networkAttribution;

    @JsonProperty("connectome_graph")
    private Map<String, Object> connectomeGraph;

    @JsonProperty("image_thumbnail_base64")
    private String imageThumbnailBase64;

    @JsonProperty("facial_landmarks")
    private List<Map<String, Object>> facialLandmarks;

    @JsonProperty("landmarks_68")
    private List<Map<String, Object>> landmarks68;

    @JsonProperty("gcn_saliency_edges")
    private List<Map<String, Object>> gcnSaliencyEdges;

    @JsonProperty("gradcam_hotspots")
    private List<Map<String, Object>> gradcamHotspots;

    @JsonProperty("biomarkers")
    private Map<String, Double> biomarkers;

    @JsonProperty("clinical_note")
    private String clinicalNote;

    @JsonProperty("created_at")
    private LocalDateTime createdAt;

    /**
     * Normalizes discrepancies between FastAPI fMRI and Image CNN schemas:
     * - Maps probability_asd / probability_typical to asdProbability / controlProbability if null
     * - Derives confidencePercentage from confidence if null
     * - Derives predictedClass and predictedLabel if either is missing
     * - Maps landmarks_68 to facial_landmarks if null
     * 
     * Why: Prevents null-pointer exceptions or Hibernate non-null constraint violations
     * when saving reports or returning unified schemas to the React frontend.
     */
    public void normalizeFields() {
        if (this.asdProbability == null && this.probabilityAsd != null) {
            this.asdProbability = this.probabilityAsd;
        }
        if (this.controlProbability == null && this.probabilityTypical != null) {
            this.controlProbability = this.probabilityTypical;
        }
        if (this.probabilityAsd == null && this.asdProbability != null) {
            this.probabilityAsd = this.asdProbability;
        }
        if (this.probabilityTypical == null && this.controlProbability != null) {
            this.probabilityTypical = this.controlProbability;
        }

        if (this.asdProbability == null) {
            this.asdProbability = 0.50;
        }
        if (this.controlProbability == null) {
            this.controlProbability = 1.0 - this.asdProbability;
        }

        if (this.predictedClass == null) {
            this.predictedClass = (this.asdProbability >= 0.50) ? 1 : 0;
        }
        if (this.predictedLabel == null || this.predictedLabel.isBlank()) {
            this.predictedLabel = (this.predictedClass == 1)
                    ? "Autism Spectrum Disorder"
                    : "Typical Control";
        }

        if (this.confidencePercentage == null) {
            if (this.confidence != null) {
                this.confidencePercentage = Math.round(this.confidence * 1000.0) / 10.0;
            } else {
                double topProb = Math.max(this.asdProbability, this.controlProbability);
                this.confidencePercentage = Math.round(topProb * 1000.0) / 10.0;
            }
        }
        if (this.confidence == null && this.confidencePercentage != null) {
            this.confidence = Math.round((this.confidencePercentage / 100.0) * 10000.0) / 10000.0;
        }

        if (this.facialLandmarks == null && this.landmarks68 != null) {
            this.facialLandmarks = this.landmarks68;
        }
        if (this.landmarks68 == null && this.facialLandmarks != null) {
            this.landmarks68 = this.facialLandmarks;
        }
    }
}
