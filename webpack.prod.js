const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyPlugin = require('copy-webpack-plugin');

module.exports = {
  entry: {
    app: './src/app.js',
    league: './src/app_league.js'
  },
  module: {
    rules: [
      { test: /\.wgsl$/i, type: 'asset/source' }
    ]
  },
  output: {
    path: path.resolve(__dirname, 'docs'),
    filename: '[name].bundle.js',
    clean: {
      // Preserve implementation tracking docs under docs/ between builds.
      keep: /NEXTGEN_.*\.md$/
    }
  },
  plugins: [
    new HtmlWebpackPlugin({
      title: 'AlphaPlague - Self-Play RL',
      filename: 'index.html',
      chunks: ['app']
    }),
    new HtmlWebpackPlugin({
      title: 'AlphaPlague — League',
      filename: 'league.html',
      chunks: ['league']
    }),
    new CopyPlugin({
      patterns: [
        { from: 'bench.html', to: 'bench.html' }
      ]
    })
  ],
  mode: 'production'
};
