// routes/interaction.js
const express = require("express");
const axios = require("axios");
const Interaction = require("../models/Interaction");

const router = express.Router();
const ML_URL = process.env.ML_URL || "http://127.0.0.1:8000/predict_enhanced";

/* In-memory KB — expand to CSV/DB in prod */
const KB = [
  { a: "paracetamol", b: "amoxicillin", interaction: "none" },
  { a: "paracetamol", b: "cetirizine", interaction: "none" },
  { a: "metformin", b: "atorvastatin", interaction: "none" },
  { a: "warfarin", b: "aspirin", interaction: "major" },
  { a: "warfarin", b: "metronidazole", interaction: "major" },
  { a: "warfarin", b: "ibuprofen", interaction: "major" },
  { a: "clopidogrel", b: "omeprazole", interaction: "moderate" },
  { a: "simvastatin", b: "clarithromycin", interaction: "major" },
  { a: "digoxin", b: "verapamil", interaction: "major" },
];

function kbCheck(drug1, drug2) {
  if (!drug1 || !drug2) return null;
  const A = String(drug1).trim().toLowerCase();
  const B = String(drug2).trim().toLowerCase();
  for (const row of KB) {
    if ((A === row.a && B === row.b) || (A === row.b && B === row.a)) {
      return row.interaction;
    }
  }
  return null;
}

function kbMetadata(kbValue) {
  if (!kbValue) return null;
  if (kbValue === "none") return { severity: "none", confRaw: 0.81, dosage_overall: "likely_safe" };
  if (kbValue === "major") return { severity: "high", confRaw: 0.95, dosage_overall: "high_risk" };
  if (kbValue === "moderate") return { severity: "moderate", confRaw: 0.85, dosage_overall: "moderate_risk" };
  return { severity: "low", confRaw: 0.7, dosage_overall: "unknown" };
}

/* Parse mg from a user-provided dosage string. Returns number (mg) or null */
function parseMgFromString(s) {
  if (!s || typeof s !== "string") return null;
  const m = s.match(/(\d+(?:\.\d+)?)\s*mg/i);
  if (!m) return null;
  const n = Number(m[1]);
  if (!Number.isFinite(n) || n <= 0) return null;
  return n;
}

/* Create readable dosage text from user or model entry */
function buildDosageTextFromUserOrModel(userDosageStr, modelEntry) {
  // prefer user dosage if provided
  if (userDosageStr && typeof userDosageStr === "string") {
    const qty = parseMgFromString(userDosageStr);
    const qtyText = qty != null ? `${qty} mg` : userDosageStr.trim();
    return { text: `${qtyText}`, qty_mg: qty, risk: null }; // risk unknown; UI will show dosage string
  }
  // fallback to model entry if provided
  if (modelEntry) {
    const qty = modelEntry.qty_mg != null ? modelEntry.qty_mg : null;
    const qtyText = qty != null ? `${qty} mg` : "unknown dose";
    return { text: `${modelEntry.drug || "Drug"}: ${qtyText} – risk: ${modelEntry.risk || "unknown"}`, qty_mg: qty, risk: modelEntry.risk || null };
  }
  return { text: "", qty_mg: null, risk: null };
}

/**
 * POST /api/interaction/check
 * body: { userId, drug1, drug2, dosageA?, dosageB? }
 */
router.post("/check", async (req, res) => {
  try {
    const { userId, drug1, drug2, dosageA, dosageB } = req.body;
    if (!drug1 || !drug2) return res.status(400).json({ message: "drug1 and drug2 are required." });

    const kb = kbCheck(drug1, drug2);
    const kbInfo = kb ? kbMetadata(kb) : null;

    // Build model input
    const text = `${drug1} ${dosageA || ""} and ${drug2} ${dosageB || ""}`.trim();

    // Always call model to get explanation bullets (but tolerate failures)
    let modelData = {};
    try {
      const mlResponse = await axios.post(ML_URL, { texts: [text] }, { timeout: 30000 });
      modelData = mlResponse.data || {};
    } catch (err) {
      console.warn("Model call failed:", err.message || err);
      modelData = {};
    }

    // Extract model outputs (safe defaults)
    const prediction = (modelData.predictions && modelData.predictions[0]) || "";
    const confScore = (modelData.confidence && modelData.confidence[0]) || null;
    const dosageInfo = (modelData.dosage_checks && modelData.dosage_checks[0]) || null;
    let explanation = (modelData.llm_explanations && modelData.llm_explanations[0]) || "";

    // Decide severity (KB deterministic override; otherwise model)
    let severity = "unknown";
    if (kbInfo) {
      severity = kbInfo.severity;
    } else {
      const predNorm = String(prediction || "").trim().toLowerCase();
      const explNorm = String(explanation || "").toLowerCase();
      const explanationSaysSafe =
        explNorm.includes("no major interaction") ||
        explNorm.includes("likely_safe") ||
        explNorm.includes("no known interaction") ||
        explNorm.includes("low risk") ||
        explNorm.includes("minimal expected");

      const confNumeric = typeof confScore === "number" ? confScore : Number(confScore) || 0;
      if (explanationSaysSafe || /no_interaction|no-interaction|no interaction|likely_safe|none/.test(predNorm)) {
        severity = "none";
      } else if (/major_interaction|major|interaction_detected|interaction/.test(predNorm)) {
        if (confNumeric >= 0.8) severity = "high";
        else if (confNumeric >= 0.6) severity = "moderate";
        else severity = "low";
      } else if (predNorm === "" || predNorm === "unknown" || predNorm === "unsure") {
        severity = "unknown";
      } else {
        if (confNumeric >= 0.8) severity = "high";
        else if (confNumeric >= 0.6) severity = "moderate";
        else if (confNumeric > 0) severity = "low";
        else severity = "unknown";
      }
    }

    // Determine final confidence: prefer model numeric confidence, else KB fallback
    let finalConfRaw = null;
    if (confScore != null && confScore !== "") {
      const asNum = Number(confScore);
      finalConfRaw = Number.isFinite(asNum) ? asNum : null;
    }
    if (finalConfRaw == null && kbInfo) finalConfRaw = kbInfo.confRaw;
    const confidencePercent = finalConfRaw != null ? Math.round(finalConfRaw * 100) : null;

    // Dosage overall: prefer KB mapping if KB present; else try to extract from explanation or dosageInfo
    let dosage_overall = kbInfo ? kbInfo.dosage_overall : null;
    if (!dosage_overall && explanation) {
      const m = explanation.match(/Overall dosage risk:\s*([\w_]+)/i);
      if (m) dosage_overall = m[1];
    }
    // If still null, infer from model dosageInfo (but we will prefer user-provided doses for display)
    if (!dosage_overall && dosageInfo && dosageInfo.aggregated && Array.isArray(dosageInfo.aggregated.per_drug)) {
      const perDrug = dosageInfo.aggregated.per_drug;
      const risks = perDrug.map((d) => (d && d.risk ? d.risk.toLowerCase() : "unknown"));
      const anyHigh = risks.some((r) => r.includes("high") || r.includes("unsafe") || r.includes("major"));
      dosage_overall = anyHigh ? "high_risk" : "likely_safe";
    }
    if (!dosage_overall) dosage_overall = "unknown";

    // Build per-drug dosage text — prefer user-provided values (parse mg)
    const userA = buildDosageTextFromUserOrModel(dosageA, (dosageInfo && dosageInfo.aggregated && dosageInfo.aggregated.per_drug && dosageInfo.aggregated.per_drug[0]) ? dosageInfo.aggregated.per_drug[0] : null);
    const userB = buildDosageTextFromUserOrModel(dosageB, (dosageInfo && dosageInfo.aggregated && dosageInfo.aggregated.per_drug && dosageInfo.aggregated.per_drug[1]) ? dosageInfo.aggregated.per_drug[1] : null);

    // Prepare final result text (KB takes precedence in phrasing)
    let resultText = "";
    if (kbInfo) {
      if (kbInfo.severity === "none") resultText = `✅ No major interaction detected between ${drug1} and ${drug2} (KB override).`;
      else if (kbInfo.severity === "high") resultText = `⚠ High concern — known major interaction between ${drug1} and ${drug2} (KB override).`;
      else if (kbInfo.severity === "moderate") resultText = `⚠ Possible moderate interaction between ${drug1} and ${drug2} (KB override).`;
      else resultText = `⚠ Possible interaction between ${drug1} and ${drug2} (KB override).`;
    } else {
      if (prediction) resultText = String(prediction);
      else {
        if (severity === "none") resultText = `✅ No major interaction detected between ${drug1} and ${drug2}`;
        else if (severity === "high") resultText = `⚠ Potential interaction detected between ${drug1} and ${drug2}`;
        else if (severity === "moderate") resultText = `⚠ Possible interaction between ${drug1} and ${drug2} (moderate confidence)`;
        else if (severity === "low") resultText = `⚠ Possible interaction (low confidence) between ${drug1} and ${drug2}`;
        else resultText = `Model is unsure about interaction between ${drug1} and ${drug2}`;
      }
    }

    // If explanation missing, synthesize clean bullets from KB
    let explanationText = explanation && String(explanation).trim();
    if (!explanationText || explanationText.length === 0) {
      if (kbInfo) {
        if (kbInfo.severity === "none") {
          explanationText =
            "- Overall dosage risk: likely_safe.\n" +
            "- Key drug details: KB indicates no known interaction.\n" +
            "- Recommendation: Combination appears low-risk; confirm with clinician if in doubt.";
        } else if (kbInfo.severity === "high") {
          explanationText =
            "- Major interaction known (see references). Increased bleeding or major PK/PD interaction possible.\n" +
            "- Recommendation: Avoid combination or monitor closely (e.g., INR monitoring for warfarin).";
        } else {
          explanationText = `- KB override: ${kb}\n- Recommendation: Verify with clinical references.`;
        }
      } else {
        explanationText = "- No explanation available from model.";
      }
    }

    // Persist log (include kb_override and prefer KB label in model_label when KB present)
    const log = await Interaction.create({
      userId,
      drug1,
      drug2,
      result: resultText,
      severity,
      model_label: kbInfo ? `KB:${kb}` : prediction || null,
      model_confidence_raw: finalConfRaw,
      model_explanation: explanationText,
      dosageA: userA ? userA.text : "",
      dosageB: userB ? userB.text : "",
      dosage_overall,
      confidence: confidencePercent,
      raw_model_response: modelData,
      kb_override: kb || null,
    });

    // Return structured response
    res.json({
      result: resultText,
      severity,
      dosageA: userA ? userA.text : "",
      dosageB: userB ? userB.text : "",
      dosage_overall,
      confidence: confidencePercent,
      explanation: explanationText,
      model_label: kbInfo ? `KB:${kb}` : prediction || null,
      logId: log._id,
    });
  } catch (err) {
    console.error("Interaction check failed:", err.message || err);
    res.status(500).json({ message: "Interaction check failed", error: err.message || "Unknown error" });
  }
});

/* GET /history */
router.get("/history/:userId", async (req, res) => {
  try {
    const logs = await Interaction.find({ userId: req.params.userId }).sort({ createdAt: -1 });
    res.json(logs);
  } catch (err) {
    console.error("History fetch failed:", err.message || err);
    res.status(500).json({ message: "History fetch failed" });
  }
});

module.exports = router;
