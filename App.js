import React, { useEffect, useState, useRef } from 'react';
import { StatusBar } from 'expo-status-bar';
import {Button, StyleSheet, Text, View} from 'react-native';

import * as tf from '@tensorflow/tfjs';
import {bundleResourceIO} from "@tensorflow/tfjs-react-native";


// Whether to load model from app bundle (true) or through network (false).
const LOAD_MODEL_FROM_BUNDLE = false;



export default function App() {
  const [tfReady, setTfReady] = useState(false);
  const [model, setModel] = useState(false);


  useEffect(() => {
    async function prepare() {

        // Wait for tfjs to initialize the backend.
       await tf.ready();

       const modelJson = require('./offline_model/model.json');
       const modelWeights1 = require('./offline_model/group1-shard1of1.bin');
       const modelUrl = bundleResourceIO(modelJson, [
           modelWeights1,
       ]);

      const model = await tf.loadLayersModel(modelUrl)
      setModel(model);

      // Ready!
      console.warn("TF now ready!")
      setTfReady(true);
    }
    prepare();
  }, []);

  const makePrediction = function(){
    if(tfReady) {
      const result = model.predict(tf.tensor2d([20], [1, 1]))
      alert(result)
    }
    else {
      console.warn("TF model not ready")
    }
  }

  return (
    <View style={styles.container}>
      <Text>Open up App.js to start working on your app!</Text>
      <StatusBar style="auto" />
      <Button title={"Predict the Future!"} onPress={makePrediction}/>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
