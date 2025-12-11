const mongoose = require("mongoose");

const InteractionSchema = new mongoose.Schema(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true
    },
    drug1: String,
    drug2: String,
    result: String,
    dosageA: String,
    dosageB: String,
    confidence: Number
  },
  { timestamps: true }
);

module.exports = mongoose.model("Interaction", InteractionSchema);
