/* eslint-disable max-len */
const functions = require("firebase-functions");
const tf = require("@tensorflow/tfjs-node");
const serviceAccount = require("./serviceAccountKey.json");
const admin = require("firebase-admin");

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
});

const express = require("express");
const cors = require("cors");

// main app
const app = express();
app.use(cors({origin: true}));

const db = admin.firestore();

// Definisikan endpoint GET
app.get("/predict", async (req, res) => {
  // Memuat model TFLite
  try {
    const modelPath = "./recommendation_model.tflite";
    const model = await tf.loadGraphModel(modelPath);

    // Ambil data dari Firestore
    const querySnapshot1 = await db.collection("megaGymDataset").get();
    const dataset1 = querySnapshot1.docs.map((doc) => doc.data());
    const querySnapshot2 = await db.collection("new_target").get();
    const dataset2 = querySnapshot2.docs.map((doc) => doc.data());
    const combinedDataset = dataset1.concat(dataset2);

    // Lakukan inferensi pada dataset
    const predictions = model.predict(tf.stack(combinedDataset));

    const input = req.query.input;
    const inputArray = input.split(",").map(Number);
    const inputShape = [1, 1966];

    // Periksa apakah inputArray memiliki bentuk yang sesuai
    if (inputArray.length !== inputShape[1]) {
      return res.status(400).send("Bentuk input tidak sesuai");
    }

    // Lakukan prediksi menggunakan model
    const inputTensor = tf.tensor(inputArray, inputShape);
    const outputTensor = model.predict(inputTensor);
    const output = outputTensor.arraySync()[0];

    // Kirim respons dengan hasil prediksi
    res.json({predictions, output});
  } catch (error) {
    console.error("Terjadi kesalahan:", error);
    res.status(500).send("Terjadi kesalahan dalam prediksi");
  }
});

// Export Firebase Function
exports.api = functions.https.onRequest(app);
