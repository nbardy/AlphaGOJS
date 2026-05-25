const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');

const faviconPath = path.resolve(__dirname, 'assets/favicon.svg');

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
    path: path.resolve(__dirname, 'dist'),
    filename: '[name].bundle.js',
    clean: true
  },
  devServer: {
    hot: true,
    client: {
      overlay: {
        errors: true,
        warnings: false
      }
    }
  },
  devtool: 'eval-source-map',
  plugins: [
    new HtmlWebpackPlugin({
      title: 'AlphaPlague',
      filename: 'index.html',
      chunks: ['app'],
      favicon: faviconPath
    }),
    new HtmlWebpackPlugin({
      title: 'AlphaPlague — League',
      filename: 'league.html',
      chunks: ['league'],
      favicon: faviconPath
    })
  ],
  mode: 'development'
};
