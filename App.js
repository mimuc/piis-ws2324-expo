import React, { useEffect, useState, useRef } from 'react';
import { StatusBar } from 'expo-status-bar';
import {Button, StyleSheet, Text, View} from 'react-native';

import * as tf from '@tensorflow/tfjs';
import {bundleResourceIO} from "@tensorflow/tfjs-react-native";



export default function App() {
  const [tfReady, setTfReady] = useState(false);
  const [model, setModel] = useState(false);


  useEffect(() => {
    async function prepare() {

        // Wait for tfjs to initialize the backend.
       await tf.ready();

       const modelJson = require('./offline_model/model.json'); // TODO I manually changed a datatype from float64 to float32
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
        const x_test = require('./demo_data/x_test.json')

        const a_values = []
        Object.keys(x_test).forEach(function(key) {
            a_values.push(x_test[key]["111"])  // true 0: 11725->0.99, 9310->0.06, 6184->0.04 ; true 1: 8536->0, 5452->0
        })
        console.log(a_values)



        const reshaped = tf.tensor2d(a_values, [1,77] )
        console.log(reshaped)

        const result = model.predict(reshaped)
        console.log(result.dataSync())

        alert(result.dataSync())
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
