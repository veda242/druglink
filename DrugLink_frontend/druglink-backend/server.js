const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
require("dotenv").config();

const authRoutes = require("./routes/auth");
const interactionRoutes = require("./routes/interaction");
const app = express();
const PORT = process.env.PORT || 5000;
const MONGO_URI =
  process.env.MONGO_URI || "mongodb://127.0.0.1:27017/druglink_db";

// Middlewares
app.use(cors({
  origin: ["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
  credentials: true
}));

app.use(express.json());

// Routes
app.use("/api/auth", authRoutes);
app.use("/api/interaction", interactionRoutes);

app.get("/", (req, res) => {
  res.send("DrugLink backend is running.");
});

// DB connection + start server
mongoose
  .connect(MONGO_URI)
  .then(() => {
    console.log("Connected to MongoDB");
    app.listen(PORT, () => {
      console.log(`DrugLink backend listening on port ${PORT}`);
    });
  })
  .catch((err) => {
    console.error("MongoDB connection error:", err);
    process.exit(1);
  });
