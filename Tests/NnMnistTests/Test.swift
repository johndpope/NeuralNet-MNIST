//
//  Test.swift
//  NnMnistTests
//
//  Created by Ilya Mikhaltsou on 06.04.2018.
//

import Foundation
import XCTest
@testable import NnMnist

class MnistTests: XCTestCase {

    @available(OSX 10.12, *)
    func testMnist() {
        do {
            let manager = try MNISTManager(
                directory: FileManager.default.temporaryDirectory.appendingPathComponent("MNIST"),
                pixelRange: (0.0, 1.0), batchSize: 100
            )
        }
        catch let error {
            XCTFail(error.localizedDescription)
        }
    }
}
