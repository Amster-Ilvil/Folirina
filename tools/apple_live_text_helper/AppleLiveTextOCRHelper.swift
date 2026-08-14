import Foundation
import VisionKit
import ImageIO
import Darwin

struct LiveTextRequest: Decodable {
    let id: String?
    let api: String?
    let image: String
    let languages: [String]?
}

struct LiveTextResponse: Encodable {
    let id: String?
    let success: Bool
    let text: String?
    let error: String?
}

func emit(_ response: LiveTextResponse) {
    do {
        let data = try JSONEncoder().encode(response)
        if let line = String(data: data, encoding: .utf8) {
            print(line)
            fflush(stdout)
        }
    } catch {
        let escaped = String(describing: error).replacingOccurrences(of: "\"", with: "'")
        print("{\"success\":false,\"error\":\"JSON encode failed: \(escaped)\"}")
        fflush(stdout)
    }
}

@available(macOS 13.0, *)
func recognizeLiveText(_ request: LiveTextRequest, analyzer: ImageAnalyzer) async -> LiveTextResponse {
    guard ImageAnalyzer.isSupported else {
        return LiveTextResponse(id: request.id, success: false, text: nil, error: "live_text_unsupported")
    }
    var configuration = ImageAnalyzer.Configuration([.text])
    if let languages = request.languages, !languages.isEmpty {
        configuration.locales = languages
    }
    do {
        let url = URL(fileURLWithPath: request.image)
        let analysis = try await analyzer.analyze(
            imageAt: url,
            orientation: .up,
            configuration: configuration
        )
        return LiveTextResponse(id: request.id, success: true, text: analysis.transcript, error: nil)
    } catch {
        return LiveTextResponse(
            id: request.id,
            success: false,
            text: nil,
            error: "VisionKit Live Text failed: \(String(describing: error))"
        )
    }
}

@main
struct AppleLiveTextOCRHelper {
    static func main() async {
        if #available(macOS 13.0, *) {
            let analyzer = ImageAnalyzer()
            while let line = readLine() {
                guard let data = line.data(using: .utf8) else { continue }
                do {
                    let request = try JSONDecoder().decode(LiveTextRequest.self, from: data)
                    guard (request.api ?? "live_text") == "live_text" else {
                        emit(LiveTextResponse(id: request.id, success: false, text: nil, error: "unsupported_api"))
                        continue
                    }
                    emit(await recognizeLiveText(request, analyzer: analyzer))
                } catch {
                    emit(LiveTextResponse(id: nil, success: false, text: nil, error: "bad_request: \(String(describing: error))"))
                }
            }
        } else {
            emit(LiveTextResponse(id: nil, success: false, text: nil, error: "live_text_requires_macos_13"))
        }
    }
}
