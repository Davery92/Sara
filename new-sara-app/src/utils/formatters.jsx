/**
 * Format message content (simple markdown-like formatting)
 * @param {string} content - The message content to format
 * @returns {string} HTML formatted content
 */
export const formatMessage = (content) => {
    if (!content) return '';
    
    // Escape HTML to prevent XSS
    let formatted = content
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
    
    // Basic markdown-like formatting
    
    // Code blocks with language
    formatted = formatted.replace(/```(\w+)?\n([\s\S]*?)\n```/g, function(match, language, code) {
      return `<pre><code class="language-${language || 'plaintext'}">${code}</code></pre>`;
    });
    
    // Inline code
    formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Bold
    formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    // Italic
    formatted = formatted.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    
    // Headers (h1, h2, h3)
    formatted = formatted.replace(/^### (.*$)/gm, '<h3>$1</h3>');
    formatted = formatted.replace(/^## (.*$)/gm, '<h2>$1</h2>');
    formatted = formatted.replace(/^# (.*$)/gm, '<h1>$1</h1>');
    
    // Convert line breaks to <br>
    formatted = formatted.replace(/\n/g, '<br>');
    
    return formatted;
  };
  
  /**
   * Clean text for TTS by removing emojis, markdown formatting, etc.
   * @param {string} text - The text to clean
   * @returns {string} Cleaned text
   */
  export const cleanTextForTTS = (text) => {
    if (!text) return '';
    
    // Remove emoji pattern
    const emojiPattern = /[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F700}-\u{1F77F}\u{1F780}-\u{1F7FF}\u{1F800}-\u{1F8FF}\u{1F900}-\u{1F9FF}\u{1FA00}-\u{1FA6F}\u{1FA70}-\u{1FAFF}\u{2702}-\u{27B0}\u{24C2}-\u{1F251}]/gu;
    let cleanedText = text.replace(emojiPattern, '');
    
    // Remove markdown formatting characters
    cleanedText = cleanedText.replace(/\*\*(.+?)\*\*/g, '$1'); // Bold: **text** to text
    cleanedText = cleanedText.replace(/\*(.+?)\*/g, '$1');     // Italic: *text* to text
    cleanedText = cleanedText.replace(/\_\_(.+?)\_\_/g, '$1'); // Underline: __text__ to text
    cleanedText = cleanedText.replace(/\_(.+?)\_/g, '$1');     // Italic with underscore: _text_ to text
    
    // Remove code blocks and inline code
    cleanedText = cleanedText.replace(/```.*?```/gs, ''); // Remove code blocks
    cleanedText = cleanedText.replace(/`(.+?)`/g, '$1');  // Inline code: `text` to text
    
    // Remove URLs
    cleanedText = cleanedText.replace(/https?:\/\/\S+/g, '');
    
    // Convert common symbols to their spoken form
    cleanedText = cleanedText.replace(/&/g, ' and ');
    cleanedText = cleanedText.replace(/%/g, ' percent ');
    cleanedText = cleanedText.replace(/\//g, ' slash ');
    cleanedText = cleanedText.replace(/=/g, ' equals ');
    
    // Remove excess whitespace and newlines
    cleanedText = cleanedText.replace(/\s+/g, ' ');
    cleanedText = cleanedText.trim();
    
    return cleanedText;
  };