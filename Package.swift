// swift-tools-version:4.0

import PackageDescription

let package = Package(
    name: "NeuralNet-MNIST",
    products: [
        .library(
            name: "NeuralNet-MNIST",
            targets: ["NnMnist"]),
    ],
    dependencies: [
        .package(url: "https://github.com/1024jp/GzipSwift.git", .upToNextMajor(from: "4.0.0"))
    ],
    targets: [
        .target(
            name: "NnMnist",
            dependencies: ["Gzip"]),
        .testTarget(
            name: "NnMnistTests",
            dependencies: ["NnMnist"]),
    ]
)
