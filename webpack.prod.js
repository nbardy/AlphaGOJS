const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const CleanWebpackPlugin = require('clean-webpack-plugin');
const webpack = require('webpack');

module.exports = {
  entry: ["babel-polyfill", './src/app.js'],
  output: {
    path: path.resolve(__dirname, 'docs'),
    filename: 'app.bundle.js'
  },
  plugins: [
    new CleanWebpackPlugin(['docs']),
    new HtmlWebpackPlugin({
      title: 'AlphaPlague - Self-Play RL'
    })
  ],
  mode: 'production',
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /(node_modules|bower_components)/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['es2017']
          }
        }
      }
    ]
  }
};
