from manim import *
import numpy as np

class FruitClassification(Scene):
    def construct(self):
        # Define colorblind-friendly colors
        COLOR_1 = "#1b9e77"  # Apples
        COLOR_2 = "#d95f02"  # Oranges
        COLOR_3 = "#7570b3"  # Unknown fruit

        # Introductory slides
        # Slide 1: Mystery fruit
        mystery_fruit = Circle(radius=1, color=COLOR_3, fill_opacity=0.5)
        question_mark = Text("?", color=WHITE).scale(2)
        mystery_group = VGroup(mystery_fruit, question_mark)
        
        title = Text("Mystery Fruit", color=WHITE).scale(0.8)
        title.to_edge(UP)
        
        self.play(Create(mystery_fruit), Write(question_mark), Write(title))
        self.wait(2)

        # Slide 2: Known fruits
        apple = Circle(radius=0.5, color=COLOR_1, fill_opacity=0.5)
        orange = Circle(radius=0.5, color=COLOR_2, fill_opacity=0.5)
        apple_text = Text("Apple", color=COLOR_1).scale(0.5).next_to(apple, DOWN)
        orange_text = Text("Orange", color=COLOR_2).scale(0.5).next_to(orange, DOWN)
        
        known_fruits = VGroup(apple, orange, apple_text, orange_text).arrange(RIGHT, buff=1)
        known_fruits.next_to(mystery_group, DOWN, buff=1)
        
        self.play(
            Create(apple),
            Create(orange),
            Write(apple_text),
            Write(orange_text)
        )
        self.wait(2)

        # Slide 3: Comparison
        arrows = VGroup(
            Arrow(start=mystery_fruit.get_bottom(), end=apple.get_top(), color=WHITE),
            Arrow(start=mystery_fruit.get_bottom(), end=orange.get_top(), color=WHITE)
        )
        
        compare_text = Text("Compare", color=WHITE).scale(0.6)
        compare_text.next_to(arrows, RIGHT)
        
        self.play(Create(arrows), Write(compare_text))
        self.wait(2)

        # Clear the introductory slides
        self.play(
            FadeOut(mystery_group),
            FadeOut(known_fruits),
            FadeOut(arrows),
            FadeOut(compare_text),
            FadeOut(title)
        )

        # Main explanation slide
        title = Text("Fruit Classification using K-Nearest Neighbors", color=WHITE).scale(0.8)
        title.to_edge(UP)
        self.play(Write(title))

        explanation = Text("We'll compare our mystery fruit to nearby fruits we already know,", 
                           color=WHITE).scale(0.6)
        explanation2 = Text("based on their size and color.", color=WHITE).scale(0.6)
        explanation.next_to(title, DOWN, buff=0.5)
        explanation2.next_to(explanation, DOWN, buff=0.2)
        self.play(Write(explanation), Write(explanation2))
        self.wait(3)

        # Clear the explanation slide
        self.play(FadeOut(title), FadeOut(explanation), FadeOut(explanation2))

        # Function to get random point within the chart
        def get_random_point():
            x = np.random.uniform(0, 10)
            y = np.random.uniform(0, 10)
            return chart.c2p(x, y)
        
        # Set up the fruit chart (size vs. color)
        chart = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            axis_config={"color": WHITE},
            tips=False,
        )
        chart_title = Text("Fruit Classification Chart", color=WHITE).scale(0.8)
        chart_title.to_edge(UP)

        # Create and position axis labels
        x_label = Text("Size", color=WHITE).scale(0.5)
        y_label = Text("Color (Redness)", color=WHITE).scale(0.5)
        x_label.next_to(chart.x_axis, DOWN)
        y_label.next_to(chart.y_axis, LEFT).rotate(90 * DEGREES)

        # Add chart and labels to the scene
        self.play(Create(chart), Write(chart_title))
        self.play(Write(x_label), Write(y_label))

        # Create fruits (data points)
        apples = VGroup(*[Dot(color=COLOR_1).move_to(chart.c2p(np.random.uniform(0, 10), np.random.uniform(0, 10))) for _ in range(15)])
        oranges = VGroup(*[Dot(color=COLOR_2).move_to(chart.c2p(np.random.uniform(0, 10), np.random.uniform(0, 10))) for _ in range(15)])

        self.play(Create(apples), Create(oranges), run_time=2)

        # Add labels for apples and oranges
        apple_label = Text("Apples", color=COLOR_1).scale(0.5)
        orange_label = Text("Oranges", color=COLOR_2).scale(0.5)
        legend = VGroup(apple_label, orange_label).arrange(RIGHT, buff=0.5)
        legend.to_corner(UR)
        self.play(Write(legend), run_time=1)


        # New fruit (point to classify)
        new_fruit = Dot(color=COLOR_3).move_to(get_random_point())
        new_fruit_label = Text("Mystery Fruit", color=COLOR_3).scale(0.5).next_to(new_fruit, DOWN)
        self.play(Create(new_fruit), Write(new_fruit_label), run_time=1)

        # Find K nearest neighbors (K=3)
        k = 3
        all_fruits = list(apples) + list(oranges)
        distances = [np.linalg.norm(fruit.get_center() - new_fruit.get_center()) for fruit in all_fruits]
        nearest_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:k]

        # Highlight nearest neighbors
        circles = VGroup(*[Circle(radius=0.3, color=COLOR_3).move_to(all_fruits[i].get_center()) 
                           for i in nearest_indices])
        self.play(Create(circles), run_time=2)

        # Count votes
        apple_votes = sum(1 for i in nearest_indices if i < len(apples))
        orange_votes = k - apple_votes

        # Show decision
        if apple_votes > orange_votes:
            decision = Text("Mystery fruit is likely an Apple!", color=COLOR_3).scale(0.7)
        else:
            decision = Text("Mystery fruit is likely an Orange!", color=COLOR_3).scale(0.7)
        decision.to_edge(DOWN)
        self.play(Write(decision), run_time=1)

        # Add a pause for emphasis
        self.wait(2)

        # Add a fun animation to show the decision
        if apple_votes > orange_votes:
            self.play(new_fruit.animate.set_color(COLOR_1).scale(1.2), run_time=1)
        else:
            self.play(new_fruit.animate.set_color(COLOR_2).scale(1.2), run_time=1)

        # Final pause
        self.wait(2)
