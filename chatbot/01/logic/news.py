from chatterbot.logic import LogicAdapter
from chatterbot.conversation import Statement
from chatterbot import languages


class News(LogicAdapter):
    """
    The News logic adapter opens news websites
    For example:
        User: 'dawaj newsy'
        Bot: 'prosz: https://www.youtube.com/watch?v=CNhtADjhQTM#ysm-group-title=Newsy'
    :kwargs:
        * *language* (``object``) --
          The language is set to ``chatterbot.languages.ENG`` for English by default.
    """

    def __init__(self, chatbot, **kwargs):
        super().__init__(chatbot, **kwargs)

        self.language = kwargs.get('language', languages.POL)
        self.cache = {}

    def can_process(self, statement):
        """
        Determines whether it is appropriate for this
        adapter to respond to the user input.
        """
        response = self.process(statement)
        self.cache[statement.text] = response
        return response.confidence == 1

    def process(self, statement, additional_response_selection_parameters=None):
        """
        Takes a statement string.
        Returns the equation from the statement with the mathematical terms solved.
        """
        # from mathparse import mathparse

        input_text = statement.text

        # Use the result cached by the process method if it exists
        if input_text in self.cache:
            cached_result = self.cache[input_text]
            self.cache = {}
            return cached_result

        # Getting the mathematical terms within the input statement
        # expression = mathparse.extract_expression(input_text, language=self.language.ISO_639.upper())

        import webbrowser
        response = Statement(text="proszsz..")

        try:
            # response.text = '{} = {}'.format(response.text)
            webbrowser.open_new_tab('https://www.youtube.com/watch?v=CNhtADjhQTM#ysm-group-title=Newsy')
            # The confidence is 1 if the expression could be evaluated
            response.confidence = 1
        except mathparse.PostfixTokenEvaluationException:
            response.confidence = 0

        return response
