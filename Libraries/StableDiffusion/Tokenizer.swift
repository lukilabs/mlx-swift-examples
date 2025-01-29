// Copyright © 2024 Apple Inc.

import Foundation

struct Bigram: Hashable {
    let a: String
    let b: String

    init(_ s: String) {
        let pieces = s.split(separator: " ")
        precondition(pieces.count == 2, "BPEPair expected two pieces for '\(s)'")
        self.a = String(pieces[0])
        self.b = String(pieces[1])
    }

    init(_ a: String, _ b: String) {
        self.a = a
        self.b = b
    }

    init(_ v: (String, String)) {
        self.a = v.0
        self.b = v.1
    }
}

/// A CLIP tokenizer.
///
/// Ported from:
///
/// - https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/tokenizer.py
/// - https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/tokenization_clip.py
///
/// Ideally this would be a tokenizer from `swift-transformers` but this is too special purpose to be representable in
/// what exists there (at time of writing).
/// Byte-Pair Encoding tokenizer for CLIP.
/// This uses an NSRegularExpression-based fallback to ensure iOS 15 compatibility.
class CLIPTokenizer {
    
    /// The NSRegularExpression pattern for splitting input text.
    ///
    /// Note that we still include `[\p{L}]+` (letters), `[\p{N}]` (single digit or numeric character),
    /// and `[^\s\p{L}\p{N}]+` (anything else).
    /// If you prefer grouping digits together, you can change `[\p{N}]` to `[\p{N}]+`.
    private let pattern = #"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"#
    
    /// Mapping from bigram -> rank (lower rank = merges earlier).
    let bpeRanks: [Bigram: Int]
    
    /// Mapping from string token -> integer token ID.
    let vocabulary: [String: Int]
    
    /// The special tokens
    let bos = "<|startoftext|>"
    let eos = "<|endoftext|>"
    
    let bosToken: Int
    let eosToken: Int
    
    /// Cache from text -> array of BPE merges to avoid recalculating
    var cache = [String: [String]]()
    
    /// Initializes the tokenizer with a list of merges and a vocabulary.
    init(merges: [String], vocabulary: [String: Int]) {
        // Convert each "merge line" into a Bigram, then rank them in the order provided
        self.bpeRanks =
            Dictionary(
                uniqueKeysWithValues:
                    merges
                    .map { Bigram($0) }
                    .enumerated()
                    .map { ($0.element, $0.offset) }
            )
        
        self.vocabulary = vocabulary
        
        // Cache the special tokens so we don’t recalculate them
        self.cache[bos] = [bos]
        self.cache[eos] = [eos]
        
        // We assume your vocabulary *must* contain the special tokens
        self.bosToken = vocabulary[bos]!
        self.eosToken = vocabulary[eos]!
    }
    
    /// Applies byte-pair encoding merges to a given text, returning merged tokens.
    func bpe(text: String) -> [String] {
        // Check if we already have a BPE result in the cache
        if let result = cache[text] {
            return result
        }
        
        precondition(!text.isEmpty, "bpe(text:) called with an empty string.")
        
        // Basic approach: keep unigrams in an array, merging from left to right
        var unigrams = text.dropLast().map { String($0) } + ["\(text.last!)</w>"]
        
        // Create a set of all adjacent bigrams in `unigrams`
        var uniqueBigrams = Set(zip(unigrams, unigrams.dropFirst()).map { Bigram($0) })
        
        // In every iteration, try to merge the two most likely bigrams. If none was merged, we’re done
        while !uniqueBigrams.isEmpty {
            // Pick the bigram with the lowest rank (bpeRanks) in `uniqueBigrams`
            let (bigram, _) = uniqueBigrams
                .map { ($0, bpeRanks[$0] ?? Int.max) }
                .min { $0.1 < $1.1 }!
            
            // If that bigram does not appear in bpeRanks, we are done
            if bpeRanks[bigram] == nil {
                break
            }
            
            var newUnigrams = [String]()
            var skip = false
            
            // Merge occurrences of the chosen bigram (a, b) into a single token `ab`
            for (a, b) in zip(unigrams, unigrams.dropFirst()) {
                if skip {
                    skip = false
                    continue
                }
                
                if Bigram(a, b) == bigram {
                    newUnigrams.append(a + b)
                    skip = true
                } else {
                    newUnigrams.append(a)
                }
            }
            
            // Append the final unigram if we didn’t merge at the end
            if !skip, let last = unigrams.last {
                newUnigrams.append(last)
            }
            
            // Update the array of unigrams and the set of bigrams
            unigrams = newUnigrams
            uniqueBigrams = Set(zip(unigrams, unigrams.dropFirst()).map { Bigram($0) })
        }
        
        // Update the cache
        cache[text] = unigrams
        return unigrams
    }
    
    /// Splits a string into token IDs (Int32), adding <bos> and <eos> tokens.
    public func tokenize(text: String) -> [Int32] {
        // Lower-case the text and collapse multiple whitespace to a single space
        let clean = text
            .lowercased()
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
        
        // Split the text according to the pattern
        // (This method uses NSRegularExpression behind the scenes.)
        let rawTokens = clean.allMatches(of: pattern)
        
        // For each raw token, apply the BPE merges
        let bpeTokens = rawTokens.flatMap { bpe(text: String($0)) }
        
        // Convert each piece into the integer token ID (if in the vocabulary)
        // and wrap with <bos> / <eos>.
        let result = [bosToken] + bpeTokens.compactMap { vocabulary[$0] } + [eosToken]
        
        // Convert everything to Int32 for typical ML runtimes
        return result.map { Int32($0) }
    }
}

/// NSRegularExpression-based helper to return all regex matches as Substrings.
extension String {
    /// Returns all non-overlapping matches of `pattern` within `self`.
    func allMatches(of pattern: String) -> [Substring] {
        do {
            let regex = try NSRegularExpression(pattern: pattern, options: [])
            let matches = regex.matches(in: self, range: NSRange(startIndex..., in: self))
            return matches.compactMap { Range($0.range, in: self).map { self[$0] } }
        } catch {
            return []
        }
    }
}
