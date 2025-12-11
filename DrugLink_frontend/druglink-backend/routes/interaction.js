const express = require("express");
const axios = require("axios");
const Interaction = require("../models/Interaction");

const router = express.Router();

const ML_URL = process.env.ML_URL || "http://127.0.0.1:8000/predict_enhanced";

/**
 * POST /api/interaction/check
 * Body: { userId, drug1, drug2, dosageA?, dosageB? }
 */
router.post("/check", async (req, res) => {
  try {
    const { userId, drug1, drug2, dosageA, dosageB } = req.body;

    if (!drug1 || !drug2) {
      return res.status(400).json({ message: "drug1 and drug2 are required." });
    }

    // Build input text for ML model
    const text = `${drug1} ${dosageA || ""} and ${drug2} ${dosageB || ""}`.trim();

    // Call Python FastAPI model
    const mlResponse = await axios.post(ML_URL, { texts: [text] });
    const data = mlResponse.data || {};

    const prediction = (data.predictions && data.predictions[0]) || "unknown";
    const confScore = (data.confidence && data.confidence[0]) || 0;
    const dosageInfo = (data.dosage_checks && data.dosage_checks[0]) || null;
    const explanation = (data.llm_explanations && data.llm_explanations[0]) || "";

    let result;
    if (String(prediction).toLowerCase().includes("interaction")) {
      result = `⚠ Potential interaction detected between ${drug1} and ${drug2}`;
    } else if (String(prediction).toLowerCase().includes("no_interaction")) {
      result = `✅ No major interaction detected between ${drug1} and ${drug2}`;
    } else {
      result = `Model is unsure about interaction between ${drug1} and ${drug2}`;
    }

    // Very simple dosage text for UI
    let dosageTextA = "";
    let dosageTextB = "";

    if (
      dosageInfo &&
      dosageInfo.aggregated &&
      Array.isArray(dosageInfo.aggregated.per_drug)
    ) {
      const perDrug = dosageInfo.aggregated.per_drug;

      if (perDrug[0]) {
        const d = perDrug[0];
        const qty = d.qty_mg != null ? `${d.qty_mg} mg` : "unknown dose";
        dosageTextA = `${d.drug}: ${qty} – risk: ${d.risk || "unknown"}`;
      }
      if (perDrug[1]) {
        const d = perDrug[1];
        const qty = d.qty_mg != null ? `${d.qty_mg} mg` : "unknown dose";
        dosageTextB = `${d.drug}: ${qty} – risk: ${d.risk || "unknown"}`;
      }
    }

    const confidencePercent = Math.round(confScore * 100);

    // Save in Mongo history
    const log = await Interaction.create({
      userId,
      drug1,
      drug2,
      result,
      dosageA: dosageTextA,
      dosageB: dosageTextB,
      confidence: confidencePercent,
    });

    res.json({
      result,
      dosageA: dosageTextA,
      dosageB: dosageTextB,
      confidence: confidencePercent,
      explanation, // LLaMA bullet explanation string
      logId: log._id,
    });
  } catch (err) {
    console.error("Interaction check failed:", err.message || err);
    res.status(500).json({
      message: "Interaction check failed",
      error: err.message || "Unknown error",
    });
  }
});

/**
 * GET /api/interaction/history/:userId
 */
router.get("/history/:userId", async (req, res) => {
  try {
    const logs = await Interaction.find({ userId: req.params.userId }).sort({
      createdAt: -1,
    });
  res.json(logs);
  } catch (err) {
    console.error("History fetch failed:", err.message || err);
    res.status(500).json({ message: "History fetch failed" });
  }
});

module.exports = router;
