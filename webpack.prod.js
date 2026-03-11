const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyPlugin = require('copy-webpack-plugin');

module.exports = {
  entry: './src/app.js',
  output: {
    path: path.resolve(__dirname, 'docs'),
    filename: 'app.bundle.js',
    clean: {
      // Preserve implementation tracking docs under docs/ between builds.
      keep: /NEXTGEN_.*\.md$/
    }
  },
  plugins: [
    new HtmlWebpackPlugin({
      title: 'AlphaPlague - Self-Play RL'
    }),
    new CopyPlugin({
      patterns: [
        { from: 'bench.html', to: 'bench.html' }
      ]
    })
  ],
  mode: 'production'
};
