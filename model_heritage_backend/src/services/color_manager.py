"""Color management for family visualization"""

class ColorPalette:
    """Manages color assignment for families"""
    
    # Predefined color palette (excluding black and white)
    COLORS = [
        '#FF0000',  # red
        '#00FF00',  # green
        '#0000FF',  # blue
        '#800080',  # purple
        '#FFA500',  # orange
        '#FFFF00',  # yellow
        '#00FFFF',  # cyan
        '#FF00FF',  # magenta
        '#FF6347',  # tomato
        '#4B0082',  # indigo
        '#32CD32',  # lime green
        '#FF1493',  # deep pink
        '#00CED1',  # dark turquoise
        '#FF4500',  # orange red
        '#9370DB',  # medium purple
        '#20B2AA',  # light sea green
        '#FF69B4',  # hot pink
        '#7B68EE',  # medium slate blue
        '#00FA9A',  # medium spring green
        '#DC143C',  # crimson
    ]
    
    def __init__(self):
        self.family_colors = {}
        self.color_index = 0
    
    def get_family_color(self, family_id: str) -> str:
        """Get or assign a color for a family"""
        if family_id not in self.family_colors:
            # Assign next color in palette, cycling if necessary
            color = self.COLORS[self.color_index % len(self.COLORS)]
            self.family_colors[family_id] = color
            self.color_index += 1
        
        return self.family_colors[family_id]
    
    def get_all_family_colors(self) -> dict:
        """Get all family color assignments"""
        return self.family_colors.copy()
    
    def reset_colors(self):
        """Reset all color assignments"""
        self.family_colors = {}
        self.color_index = 0


# Global instance
color_palette = ColorPalette()